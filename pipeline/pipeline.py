from abc import ABC, abstractmethod
from pipeline.registry import registry
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from pipeline.pipeline_mixin import *
import numpy as np
from model.vision.basic_modules import generate_causal_mask
import sys
from tqdm import tqdm
import os
from utils.analysis import summarize_pipeline
import torch.profiler as tprof
'''
Base class for all pipelines
'''
class Pipeline(ABC):
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def end(self):
        pass
    

    def run_all(self):
        self.initialize()
        self.run()
        self.end()

class OptimusPrimePipeline(Pipeline, NormalDataloaderMixin, ModelOptimizationMixin, ModelEvaluationMixin, ModelMetricMixin, ModelLossMixin):
    def __init__(self, cfg):
        # build saver and logger
        if not cfg['eval_task']:
            self.logger = registry.get_utils(cfg['logger']['name'])(cfg)
        self.saver = registry.get_utils(cfg['saver']['name'])(**cfg['saver']['args'])
       
        # build model
        self.lang_encoder = registry.get_language_model(cfg['lang_encoder']['name'])(**cfg['lang_encoder']['args']).cuda()
        self.point_encoder = registry.get_vision_model(cfg['point_encoder']['name'])(**cfg['point_encoder']['args']).cuda()
        self.unified_encoder = registry.get_vision_model(cfg['unified_encoder']['name'])(**cfg['unified_encoder']['args']).cuda()
        self.ground_head = registry.get_other_model(cfg['ground_head']['name'])(**cfg['ground_head']['args']).cuda()
        self.qa_head = registry.get_other_model(cfg['qa_head']['name'])(**cfg['qa_head']['args']).cuda()
        self.pretrain_head = registry.get_other_model(cfg['pretrain_head']['name'])(**cfg['pretrain_head']['args']).cuda()
        self.caption_head = registry.get_other_model(cfg['caption_head']['name'])(**cfg['caption_head']['args']).cuda()
        
        # ours
        if cfg['task'] == 'coarse_ground' or cfg['task'] == 'scanqa':
            self.cghead = registry.get_other_model(cfg['coarse_ground_head']['name'])(**cfg['coarse_ground_head']['args']).cuda()
        if cfg['task'] == 'fine_ground' or cfg['task'] == 'scanqa':
            self.fghead = registry.get_other_model(cfg['fine_ground_head']['name'])(**cfg['fine_ground_head']['args']).cuda()
        if cfg['task'] == 'inference' or cfg['task'] == 'scanqa':
            self.inference_head = registry.get_other_model(cfg['inference_head']['name'])(**cfg['inference_head']['args']).cuda()
        if cfg['task'] == 'scanqa':
            self.suphead = registry.get_other_model(cfg['supervise_base_head']['name'])(**cfg['supervise_base_head']['args']).cuda()
        # load task
        self.task = cfg['task']
        self.eval_task = cfg['eval_task']
        
        print("\n" + "="*50)
        print(f"[INFO] Pipeline initialized for task: '{self.task}'")
        print(f"[INFO] Evaluation mode (eval_task): {self.eval_task}")
        if cfg.get('restore_model'):
            load_dir = cfg['saver']['args'].get('load_dir')
            load_name = cfg['saver']['args'].get('load_name')
            if load_dir and load_name:
                # Reconstruct the path exactly as ModelSaver does
                full_path = os.path.join(load_dir, load_name)
                print(f"[INFO] Restore checkpoint path specified: {full_path}")
            else:
                print("[WARNING] 'restore_model' is True, but 'load_dir' or 'load_name' is missing in saver config.")
        else:
            print("[INFO] 'restore_model' is False. Training from scratch or without loading a checkpoint.")
        print("="*50 + "\n")
        
        # build dataset
        if self.task == 'scanrefer' or self.task == 'referit3d':
            self.train_dataset = registry.get_dataset(cfg['refer_dataset']['name'])(split='train', **cfg['refer_dataset']['args'])
            self.test_dataset = registry.get_dataset(cfg['refer_dataset']['name'])(split='val', **cfg['refer_dataset']['args'])
        elif self.task == 'scanqa' or self.task == 'sqa':
            # only qa has test dataset
            if cfg['qa_dataset']['args'].get('split') != None:
                assert self.eval_task == True
                self.train_dataset = registry.get_dataset(cfg['qa_dataset']['name'])(**cfg['qa_dataset']['args'])
                self.test_dataset = registry.get_dataset(cfg['qa_dataset']['name'])(**cfg['qa_dataset']['args'])
            else:
                self.train_dataset = registry.get_dataset(cfg['qa_dataset']['name'])(split='train', **cfg['qa_dataset']['args'])
                self.test_dataset = registry.get_dataset(cfg['qa_dataset']['name'])(split='val', **cfg['qa_dataset']['args'])
        elif self.task == 'caption':
            if cfg['caption_dataset']['args'].get('split') != None:
                assert self.eval_task == True
                self.train_dataset = registry.get_dataset(cfg['caption_dataset']['name'])(**cfg['caption_dataset']['args'])
                self.test_dataset = registry.get_dataset(cfg['caption_dataset']['name'])(**cfg['caption_dataset']['args'])
            else:
                self.train_dataset = registry.get_dataset(cfg['caption_dataset']['name'])(split='train', **cfg['caption_dataset']['args'])
                self.test_dataset = registry.get_dataset(cfg['caption_dataset']['name'])(split='val', **cfg['caption_dataset']['args'])
        else:
            raise NotImplementedError("task " + self.task + " is not implemented")
      
        # load optimize config
        self.batch_size = cfg['batch_size']
        self.learning_rate = cfg['learning_rate']
        self.grad_norm = cfg['grad_norm']
        self.epochs = cfg['epochs']
        self.warmup_steps = cfg['warmup_steps']
        
        # build dataloaer
        self.build_train_test_dataloader()
        
        # add parameters
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += self.no_decay_param_group(self.lang_encoder.named_parameters(), self.learning_rate * cfg['lang_lr_mul'])
        optimizer_grouped_parameters += self.no_decay_param_group(self.point_encoder.named_parameters(), self.learning_rate * cfg['point_lr_mul'])
        optimizer_grouped_parameters += self.no_decay_param_group(self.unified_encoder.named_parameters(), self.learning_rate * cfg['unified_lr_mul'])
        optimizer_grouped_parameters += self.no_decay_param_group(self.ground_head.named_parameters(), self.learning_rate)
        optimizer_grouped_parameters += self.no_decay_param_group(self.qa_head.named_parameters(), self.learning_rate)
        optimizer_grouped_parameters += self.no_decay_param_group(self.pretrain_head.named_parameters(), self.learning_rate)
        optimizer_grouped_parameters += self.no_decay_param_group(self.caption_head.named_parameters(), self.learning_rate)
        if self.task == 'coarse_ground' or self.task == 'scanqa':
            optimizer_grouped_parameters += self.no_decay_param_group(self.cghead.named_parameters(), self.learning_rate)
        if self.task == 'fine_ground' or self.task == 'scanqa':
            optimizer_grouped_parameters += self.no_decay_param_group(self.fghead.named_parameters(), self.learning_rate)
        if self.task == 'inference' or self.task == 'scanqa':
            optimizer_grouped_parameters += self.no_decay_param_group(self.inference_head.named_parameters(), self.learning_rate)
        if self.task == 'scanqa':
            optimizer_grouped_parameters += self.no_decay_param_group(self.suphead.named_parameters(), self.learning_rate)
        # build optimizer
        self.optimizer = AdamW(optimizer_grouped_parameters, betas=[cfg['beta1'], cfg['beta2']])
        self.parameters = []
        for p in optimizer_grouped_parameters:
            self.parameters.extend(p['params'])
        
        # build scheduler
        total_steps = self.epochs * len(self.train_data_loader)
        self.total_steps = total_steps
        print("total_steps {}".format(total_steps))
        # valid_steps = 30 * len(self.train_data_loader)
        lambda_warmup_consine = lambda step: self.warmup_cosine(step, self.warmup_steps, total_steps)
        self.scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lambda_warmup_consine)
       
        # build loss function
        if self.task == 'scanrefer' or self.task == 'referit3d':
            self.refer_loss = registry.get_optimizer(cfg["refer_loss"]['name'])
        elif self.task == 'scanqa' or self.task == 'sqa':
            self.qa_loss = registry.get_optimizer(cfg["qa_loss"]['name'])
        elif self.task == 'caption':
            self.caption_loss = registry.get_optimizer(cfg['caption_loss']['name'])
        
        # restore model
        if cfg['restore_model']:
            self.restore_model()
        self.total_params, self.trainable_params = summarize_pipeline(self)
        self._profiled_forward = False
        
    def initialize(self):
        pass
    

    @torch.no_grad()
    def _profile_once(self, data_dict, tag='eval'):
        """
        Profile one real forward pass on *data_dict*.
        Prints GFLOPs/GMACs & 可选日志记录。
        """
        if self._profiled_forward:        # 只跑一次
            return

        activities = [tprof.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(tprof.ProfilerActivity.CUDA)

        with tprof.profile(
            activities=activities,
            with_flops=True,          # 让 PyTorch 自动算 FLOPs
            record_shapes=False,
            profile_memory=False
        ) as prof:
            _ = self.forward_one(data_dict)   # 真正前向，**用的就是 eval batch**

        # 统计
        avg = prof.key_averages().total_average()
        flops = avg.flops                # 单次前向总 FLOPs
        macs  = flops / 2                # 通常 1 MAC = 2 FLOPs

        gflops = flops / 1e9
        gmacs  = macs  / 1e9
        print(f"[Profiler-{tag}] ≈ {gmacs:.2f} GMACs ({gflops:.2f} GFLOPs) per forward")

        # 如果你在用 self.logger，可顺手记录
        if hasattr(self, "logger"):
            self.logger.log({f'{tag}/GMACs': gmacs}, step=0)

        self._profiled_forward = True

    '''
    Hierarchical
    train
        forward
        get loss
        get metric
        record train
    eval
        forward
        get metric
        process metric
        record eval
    '''    
    def run(self):
        best_target_metric = -np.inf # assuming higher is better
        for epoch in range(self.epochs):
            # direct eval
            if self.eval_task:
                print("[INFO] 'eval_task' is True. Running evaluation only and then exiting.")
                self.eval(epoch)
                break
            # train
            self.train(epoch)
            # eval
            if self.task != 'caption':
                target_metric = self.eval(epoch)
                if target_metric > best_target_metric:
                    best_target_metric = target_metric
                    self.save_model()
            else:
                self.save_model() # for pretrain and caption save directly

    def train(self, epoch):
        print(f"\nTraining Epoch {epoch:02} ...")
        self.set_model_state('train')
        tqdm_out = sys.stderr if sys.stderr.isatty() else open(os.devnull, 'w')
        for i, data_dict in enumerate(tqdm(self.train_data_loader, file=tqdm_out)):
            # add step and total steps to data_dict
            data_dict['cur_step'] = epoch * len(self.train_data_loader) + i 
            data_dict['total_steps'] = self.total_steps
            
            # forward
            data_dict = self.forward_one(data_dict)
        
            # calculate loss
            data_dict = self.get_loss(data_dict)
            
            # calculate metrics
            data_dict = self.get_metrics(data_dict)
            
            # optimize
            loss = data_dict['total_loss']
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters, self.grad_norm
            )
            data_dict['grad_norm'] = grad_norm
            self.optimizer.step()
            self.optimizer.zero_grad()     
            self.scheduler.step()
            
            # record
            step = epoch * len(self.train_data_loader) + i 
            self.record_train_step(data_dict, step)
            
    def eval(self, epoch):
        print("start evaluation on test set")
        self.set_model_state('eval')
        # build eval dict
        if self.task == 'scanrefer':
            return self.eval_scanrefer(epoch)
        elif self.task == 'referit3d':
            return self.eval_referit3d(epoch)
        elif self.task == 'scanqa':
            return self.eval_qa(epoch)
        elif self.task == 'sqa':
            return self.eval_sqa(epoch)
        elif self.task == 'caption':
            return self.eval_caption(epoch)
               
    def forward_one(self, data_dict):
        # prepare data
        self.prepare_data(data_dict)
        
        # prepare dict
        if 'cur_step' not in data_dict.keys():
            data_dict['cur_step'] = 1
            data_dict['total_steps'] = 1
        # basic feature extracter
        # point_features_pre_spatial is point features before spatial reasonging
        if self.task == 'caption':
            causal_mask = generate_causal_mask(data_dict['txt_ids'].shape[1]).unsqueeze(0).repeat(data_dict['txt_ids'].shape[0], 1, 1).cuda()
            lang_basic_features = self.lang_encoder(data_dict['txt_ids'], causal_mask).last_hidden_state
        else:
            lang_basic_features = self.lang_encoder(data_dict['txt_ids'], data_dict['txt_masks']).last_hidden_state
        point_basic_features, point_features_pre, obj_cls_raw_logits = self.point_encoder(data_dict['obj_fts'].float(), data_dict['obj_locs'], data_dict['obj_masks'], data_dict['obj_sem_masks'], 
                                                                                          data_dict['obj_labels'], data_dict['cur_step'], data_dict['total_steps'])
        
        # unifed language entity transformer
        if self.task == 'caption':
            language_fuse_feature, point_fuse_feature  = self.unified_encoder(lang_basic_features, data_dict['txt_masks'], point_basic_features, data_dict['obj_locs'], data_dict['obj_masks'], data_dict['tgt_object_id'], True)
        else:
            language_fuse_feature, point_fuse_feature  = self.unified_encoder(lang_basic_features, data_dict['txt_masks'], point_basic_features, data_dict['obj_locs'], data_dict['obj_masks'])
        
        # task head
        txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, og3d_logits = self.ground_head(language_fuse_feature, point_fuse_feature, point_features_pre, data_dict['obj_masks'])
        
        # not implemented tasks: coarse_ground, fine_ground, inference
        #now cancelling hcnqa
        if self.task == 'scanqa':
            # 0. Pre-HSM Feature Extraction:
            #    - Input: point_fuse_feature (fused object features 'o' from unified_encoder)
            #    - Module: self.suphead (MLP layers)
            #    - Output: sup_embeds (transformed base features 'F_base' for HSM stages)
            sup_embeds = self.suphead(point_fuse_feature)
            
            # 1. Coarse Grounding (CG) Stage (Corresponds to BoI - Block of Interest):
            #    - Input: sup_embeds (F_base)
            #    - Module: self.cghead (CoarseGroundHeadV4)
            #    - Outputs:
            #        - cg_res: Refined features after CG stage ('F_cg'), passed to the next stage (FG).
            #        - cg_logits: Raw prediction scores for each object being in BoI. Used to calculate Weighted BCE Loss for this stage.
            #        - cg_mask: Sigmoid probabilities.
            cg_res, cg_logits, cg_mask = self.cghead(sup_embeds)
            data_dict['cg_mask'] = cg_mask
            data_dict['cg_logits'] = cg_logits
            
            # 2. Fine Grounding (FG) Stage (Corresponds to OoI - Objects of Interest):
            #    - Inputs:
            #        - sup_embeds (F_base): Original base features.
            #        - cg_res (F_cg): Features refined by the previous CG stage.
            #    - Module: self.fghead (FineGroundHeadV4) - internally concatenates inputs.
            #    - Outputs:
            #        - fg_res: Refined features after FG stage ('F_fg'), passed to the next stage (Inference).
            #        - fg_logits: Raw prediction scores for each object being in OoI. Used for L_fg loss.
            #        - fg_mask: Sigmoid probabilities.
            fg_res, fg_logits, fg_mask = self.fghead(sup_embeds, cg_res)
            data_dict['fg_mask'] = fg_mask
            data_dict['fg_logits'] = fg_logits
            
             # 3. Inference (Infer) Stage (Corresponds to OoT - Object of Target):
            #    - Inputs:
            #        - sup_embeds (F_base): Original base features.
            #        - cg_res (F_cg): Features from CG stage (passed, though not used internally by InferenceHeadV4).
            #        - fg_res (F_fg): Features refined by the previous FG stage.
            #    - Module: self.inference_head (InferenceHeadV4) - internally concatenates sup_embeds and fg_res.
            #    - Outputs:
            #        - infer_att_mask ('att_out' in InferenceHeadV4): The crucial attention modulation signal ('M_if') passed to the VQA head.
            #        - infer_logits: Raw prediction scores for each object being the OoT. Used for L_if loss.
            #        - infer_mask: Sigmoid probabilities.
            infer_att_mask, infer_logits, infer_mask = self.inference_head(sup_embeds, cg_res, fg_res)
            data_dict['infer_mask'] = infer_mask
            data_dict['infer_logits'] = infer_logits

        
        if self.task == 'scanqa':
            # For the ScanQA task (HCNQA), call the VQA head with the attention mask from the HSM.
            # Inputs to QA Head:
            #    - point_fuse_feature: Original fused object features ('o').
            #    - language_fuse_feature: Original fused language features.
            #    - obj_masks, txt_masks: Padding masks.
            #    - infer_att_mask: The attention modulation signal ('M_if') from the Inference stage.
            #    - 'mul': Specifies the multiplicative modulation method (implements o' = o * (M + 1) concept).
            answer_scores = self.qa_head(point_fuse_feature, data_dict['obj_masks'], language_fuse_feature, data_dict['txt_masks'], infer_att_mask, 'mul')
            # answer_scores = self.qa_head(point_fuse_feature, data_dict['obj_masks'], language_fuse_feature, data_dict['txt_masks'])
        else:
            answer_scores = self.qa_head(point_fuse_feature, data_dict['obj_masks'], language_fuse_feature, data_dict['txt_masks'])
        txt_lm_cls_logits, scene_txt_match_logit = self.pretrain_head(language_fuse_feature)
        txt_caption_cls_logit = self.caption_head(language_fuse_feature)
        
        data_dict['txt_cls_logits'] = txt_cls_logits
        data_dict['obj_cls_post_logits'] = obj_cls_post_logits
        data_dict['obj_cls_pre_logits'] = obj_cls_pre_logits
        data_dict['obj_cls_raw_logits'] = obj_cls_raw_logits
        data_dict['og3d_logits'] = og3d_logits
        data_dict['answer_scores'] = answer_scores
        data_dict['txt_lm_cls_logits'] = txt_lm_cls_logits
        data_dict['scene_txt_match_logit'] = scene_txt_match_logit
        data_dict['txt_caption_cls_logit'] = txt_caption_cls_logit
        
        return data_dict
    
    def get_loss(self, data_dict):
        if self.task == 'scanrefer' or self.task == 'referit3d' :
            data_dict = self.get_refer_loss(data_dict)
        elif self.task == 'scanqa':
            data_dict = self.get_qa_loss(data_dict)
        elif self.task == 'sqa':
            data_dict = self.get_sqa_loss(data_dict)
        elif self.task == 'caption':
            data_dict = self.get_caption_loss(data_dict)
        return data_dict
    
    def get_metrics(self, data_dict):
        if self.task == 'scanrefer':
            data_dict = self.get_scanrefer_metrics(data_dict)
        elif self.task == 'referit3d':
            data_dict = self.get_referit3d_metrics(data_dict)
        elif self.task == 'scanqa':
            data_dict = self.get_qa_metrics(data_dict)
        elif self.task == 'sqa':
            data_dict = self.get_sqa_metrics(data_dict)
        elif self.task == 'caption':
            data_dict = self.get_caption_metrics(data_dict)
        return data_dict
     
    def record_train_step(self, data_dict, step):
        log_dict = {
            # basic info
            'step': step,
            'lr': self.scheduler.get_last_lr()[0],
            'grad_norm': data_dict['grad_norm'].item(),
            # shared loss
            'total_loss': data_dict['total_loss'].item(),
            'obj_cls_raw_loss': data_dict['obj_cls_raw_loss'].item(),
            'obj_cls_pre_loss': data_dict['obj_cls_pre_loss'].item(),
            'obj_cls_post_loss': data_dict['obj_cls_post_loss'].item(),
            # shared acc
            'obj_cls_raw_acc': data_dict['obj_cls_raw_acc'],
            'obj_cls_pre_acc': data_dict['obj_cls_pre_acc'],
            'obj_cls_post_acc': data_dict['obj_cls_post_acc'],
        }
        if self.task == 'scanrefer' or self.task == 'referit3d':
            log_dict.update({
                # loss
                'og3d_loss': data_dict['og3d_loss'].item(),
                'txt_cls_loss': data_dict['txt_cls_loss'].item(),
                # acc
                'og_acc': data_dict['og_acc'],
                'txt_acc': data_dict['txt_acc'],
            })
        elif self.task == 'scanqa':
            log_dict.update({
                # loss
                'og3d_loss': data_dict['og3d_loss'].item(),
                'txt_cls_loss': data_dict['txt_cls_loss'].item(),
                'ans_loss': data_dict['answer_loss'].item(),
                # acc
                'og_acc': data_dict['og_acc'],
                'txt_acc': data_dict['txt_acc'],
                'ans1_acc': data_dict['ans1_acc'],
                'ans10_acc': data_dict['ans10_acc']
            })
        elif self.task == 'sqa':
            log_dict.update({
                # loss
                'ans_loss': data_dict['answer_loss'].item(),
                # acc
                'ans1_acc': data_dict['ans1_acc'],
                'ans10_acc': data_dict['ans10_acc']
            })
        elif self.task == 'caption':
            log_dict.update({
                # loss
                'caption_cls_loss': data_dict['caption_cls_loss'],
                # acc
                'caption_cls_acc_mask': data_dict['caption_cls_acc_mask']
            })
        for k in list(log_dict.keys()):
            log_dict['train/' + k] = log_dict.pop(k)
        self.logger.log(log_dict, step=step)
            
    def record_eval_step(self, eval_dict, epoch):
        print("")
        print(f"-------------------- Epoch {epoch:02} --------------------")
        for key in eval_dict.keys():
            if self.eval_task:
                print('[test] ' + key, eval_dict[key])
            else:
                print('[test] ' + key, eval_dict[key])
                self.logger.log({'test/' + key: eval_dict[key]}, step = (epoch + 1) * len(self.train_data_loader))
        print("--------------------------------------------------")
    
    def partial_load(self, model, checkpoint):
        model_dict = model.state_dict()
        pretrained_dict = checkpoint

        compatible_dict = {}

        for k in pretrained_dict:
            if k in model_dict:
                if pretrained_dict[k].size() == model_dict[k].size():
                    compatible_dict[k] = pretrained_dict[k]
                else:
                    print(f"Size mismatch for {k}: checkpoint size {pretrained_dict[k].size()}, model size {model_dict[k].size()}")
                    if pretrained_dict[k].size(1) < model_dict[k].size(1):
                        temp_tensor = torch.zeros_like(model_dict[k])
                        temp_tensor[:, :pretrained_dict[k].size(1), :, :] = pretrained_dict[k]
                        compatible_dict[k] = temp_tensor
                    elif pretrained_dict[k].size(1) > model_dict[k].size(1):
                        compatible_dict[k] = pretrained_dict[k][:, :model_dict[k].size(1), :, :]
            else:
                print(f"Key {k} not found in model.")
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict, strict=False)
    
    def load_encoder(self, state_dict):
        try:
            self.point_encoder.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Failed to load state dict: {e}")
            self.partial_load(self.point_encoder, state_dict)
    
    def load_qa_head(self, state_dict):
        try:
            self.qa_head.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Failed to load state dict: {e}")
            self.partial_load(self.qa_head, state_dict)
    
    def restore_model(self):
        print("\n" + "="*50)
        print("[INFO] Attempting to restore model from checkpoint...")
        state_dict = self.saver.restore_dict()
        if state_dict is None:
            print("[ERROR] Failed to restore checkpoint: self.saver.restore_dict() returned None.")
            print("="*50 + "\n")
            return
        print("[SUCCESS] Checkpoint dictionary loaded from saver.")
        
        # --- Core Components ---
        try:
            self.lang_encoder.load_state_dict(state_dict['lang_encoder'])
            print("[SUCCESS] Loaded 'lang_encoder' weights.")
        except Exception as e:
            print(f"[ERROR] Failed to load 'lang_encoder': {e}")
            
        # self.point_encoder.load_state_dict(state_dict['point_encoder'])
        try:
            self.load_encoder(state_dict['point_encoder']) # This calls partial_load
            print("[SUCCESS] Loaded 'point_encoder' weights (using partial load).")
        except Exception as e:
            print(f"[ERROR] Failed to load 'point_encoder': {e}")
            
        try:
            self.unified_encoder.load_state_dict(state_dict['unified_encoder'])
            print("[SUCCESS] Loaded 'unified_encoder' weights.")
        except Exception as e:
            print(f"[ERROR] Failed to load 'unified_encoder': {e}")
            
        try:
            self.ground_head.load_state_dict(state_dict['ground_head'])
            print("[SUCCESS] Loaded 'ground_head' weights.")
        except Exception as e:
            print(f"[ERROR] Failed to load 'ground_head': {e}")
            
        # try:
        #     self.qa_head.load_state_dict(state_dict['qa_head'])
        # except: 
        #     print("fail to load qa params")
        try:
            self.load_qa_head(state_dict['qa_head'])
            print("[SUCCESS] Loaded 'qa_head' weights (using partial load).")
        except Exception as e:
            print(f"[ERROR] Failed to load 'qa_head': {e}")
            
        try:
            self.pretrain_head.load_state_dict(state_dict['pretrain_head'])
            print("[SUCCESS] Loaded 'pretrain_head' weights.")
        except Exception as e:
            print(f"[ERROR] Failed to load 'pretrain_head': {e}")

        try:
            self.caption_head.load_state_dict(state_dict['caption_head'])
            print("[SUCCESS] Loaded 'caption_head' weights.")
        except:
            print("[WARNING] Skipped loading 'caption_head'. This may be expected.")
            
        try:
            if(self.task == 'coarse_ground' or self.task == 'scanqa'): 
                self.cghead.load_state_dict(state_dict['coarse_ground_head'])
                print("[SUCCESS] Loaded 'coarse_ground_head' weights.")
        except:
            print("fail to load cg params")
        try:
            if(self.task == 'fine_ground' or self.task == 'scanqa'):
                self.fghead.load_state_dict(state_dict['fine_ground_head'])
                print("[SUCCESS] Loaded 'fine_ground_head' weights.")
        except:
            print("fail to load fg params")
        try:
            if(self.task == 'inference' or self.task == 'scanqa'):
                self.inference_head.load_state_dict(state_dict['inference_head'])
                print("[SUCCESS] Loaded 'inference_head' weights.")
        except:
            print("fail to load infer params")
        try:
            if(self.task == 'scanqa'):
                self.suphead.load_state_dict(state_dict['supervise_base_head'])
                print("[SUCCESS] Loaded 'supervise_base_head' weights.")
        except:
            print("fail to load sup_base params")
            
        print("[INFO] Model restoration process finished.")
        print("="*50 + "\n")

    def save_model(self):
        save_dict = {
            'lang_encoder': self.lang_encoder.state_dict(),
            'point_encoder': self.point_encoder.state_dict(),
            'unified_encoder': self.unified_encoder.state_dict(),
            'ground_head': self.ground_head.state_dict(),
            'qa_head': self.qa_head.state_dict(),
            'pretrain_head': self.pretrain_head.state_dict(),
            'caption_head': self.caption_head.state_dict()
        }
        # not implemented tasks: coarse_ground, fine_ground, inference
        if self.task == 'scanqa':
            save_dict['coarse_ground_head'] = self.cghead.state_dict()
            save_dict['fine_ground_head'] = self.fghead.state_dict()
            save_dict['inference_head'] = self.inference_head.state_dict()
            save_dict['supervise_base_head'] = self.suphead.state_dict()
        self.saver.save_dict(save_dict)
    
    def set_model_state(self, state='train'):
        assert state in ['train', 'eval']
        torch.cuda.empty_cache()
        if state == 'train':
            self.lang_encoder.train()
            self.point_encoder.train()
            self.unified_encoder.train()
            self.ground_head.train()
            self.qa_head.train()
            self.pretrain_head.train()
            self.caption_head.train()
            if self.task == 'scanqa':
                self.cghead.train()
                self.fghead.train()
                self.inference_head.train()
                self.suphead.train()
        else:
            self.lang_encoder.eval()
            self.point_encoder.eval()
            self.unified_encoder.eval()
            self.ground_head.eval()
            self.qa_head.eval()
            self.pretrain_head.eval()
            self.caption_head.eval()
            if self.task == 'scanqa':
                self.cghead.eval()
                self.fghead.eval()
                self.inference_head.eval()
                self.suphead.eval()
                 
    def end(self):
        pass
