import torch
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from pipeline.registry import registry


@registry.register_optimizer("refer_loss_v1")
def get_refer_loss_v1(txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, og3d_logits, tgt_object_label, tgt_object_id, obj_labels, obj_masks):
    og3d_loss = F.cross_entropy(og3d_logits, tgt_object_id.squeeze(1))
    txt_cls_loss = F.cross_entropy(txt_cls_logits, tgt_object_label.squeeze(1))
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    total_loss = og3d_loss + txt_cls_loss + obj_cls_raw_loss + obj_cls_pre_loss + obj_cls_post_loss
    return total_loss, og3d_loss, txt_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss

@registry.register_optimizer("qa_loss_v1")
def get_qa_loss_v1(txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, og3d_logits,
                   answer_scores, tgt_object_label, tgt_object_id, obj_labels, obj_masks, answer_label,
                   cg_mask, cg_gt, cg_logits, fg_mask, fg_gt, fg_logits, infer_mask, infer_gt, infer_logits):
    txt_cls_loss = F.binary_cross_entropy_with_logits(txt_cls_logits, tgt_object_label.float(), reduction='sum') / float(tgt_object_label.shape[0])
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    answer_loss = F.binary_cross_entropy_with_logits(answer_scores, answer_label.float(), reduction='sum') / answer_scores.shape[0]
    
    # ours
    cg_mask = cg_mask.to(device='cuda') # mask了问题对应场景包含的有效对象
    cg_gt = cg_gt.to(device='cuda') # 问题对应gt
    valid_cg_gt = cg_gt[obj_masks] # 只保留场景中有效对象的gt
    # 统计问题中 [ 包含的场景物体数量 / 不包含的数量 ]
    class_counts = torch.tensor([(valid_cg_gt == 0).sum(), (valid_cg_gt == 1).sum()], dtype=torch.float32, device=cg_mask.device) 
    class_counts[class_counts == 0] = 1 # 避免除0
    total = class_counts.sum() # 统计问题所在场景中的物体数量
    weights = total / class_counts # 计算 [包含在问题中物体/不包含的物体] 的权重，与物体数量成反比
    cg_weights = weights[cg_gt.long()] * obj_masks # 将权重应用到有效物体上
    # 计算交叉熵损失
    cg_loss = F.binary_cross_entropy_with_logits(cg_logits, cg_gt.float(), weight=cg_weights, reduction='sum') / float(cg_gt.shape[0])
    
    fg_mask = fg_mask.to(device='cuda')
    fg_gt = fg_gt.to(device='cuda')
    valid_fg_gt = fg_gt[obj_masks]
    class_counts = torch.tensor([(valid_fg_gt == 0).sum(), (valid_fg_gt == 1).sum()], dtype=torch.float32, device=fg_mask.device)
    class_counts[class_counts == 0] = 1
    total = class_counts.sum()
    weights = total / class_counts
    fg_weights = weights[fg_gt.long()]
    fg_loss = F.binary_cross_entropy_with_logits(fg_logits, fg_gt.float(), weight=fg_weights, reduction='sum') / float(fg_gt.shape[0])

    infer_mask = infer_mask.to(device='cuda')
    infer_gt = infer_gt.to(device='cuda')
    valid_infer_gt = infer_gt[obj_masks]
    class_counts = torch.tensor([(valid_infer_gt == 0).sum(), (valid_infer_gt == 1).sum()], dtype=torch.float32, device=infer_mask.device)
    class_counts[class_counts == 0] = 1
    total = class_counts.sum()
    weights = total / class_counts
    inf_weight = weights[infer_gt.long()] * obj_masks
    infer_loss = F.binary_cross_entropy_with_logits(infer_logits, infer_gt.float(), weight=inf_weight, reduction='sum') / float(infer_gt.shape[0])

    # og3d_loss = cg_loss * 0.2 + fg_loss * 0.3 + infer_loss * 0.5
    # og3d_loss = cg_loss * 0 + fg_loss * 0.3 + infer_loss * 0
    og3d_loss = cg_loss * 0.2 + fg_loss * 0 + infer_loss * 0.5
    total_loss = og3d_loss + txt_cls_loss + obj_cls_raw_loss + obj_cls_pre_loss + obj_cls_post_loss + answer_loss
    return total_loss, og3d_loss, txt_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, answer_loss, cg_loss, fg_loss, infer_loss

@registry.register_optimizer("pretrain_loss_v1")
def get_pretrain_loss_v1(txt_lm_cls_logits, masked_lm_labels, scene_txt_match_logits, replace, obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, obj_labels, obj_sem_masks, obj_masks):
    loss_fct = CrossEntropyLoss(ignore_index=-1)
    masked_lm_labels.masked_fill_(replace.unsqueeze(1), -1)
    lm_cls_loss = loss_fct(txt_lm_cls_logits.permute(0, 2, 1), masked_lm_labels)
    match_loss = loss_fct(scene_txt_match_logits, replace.long())
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss_mask = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks * obj_sem_masks.logical_not()).sum() / (obj_masks * obj_sem_masks.logical_not()).sum()
    obj_cls_pre_loss_unmask = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks * obj_sem_masks).sum() / (obj_masks * obj_sem_masks).sum()
    obj_cls_post_loss_mask = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks * obj_sem_masks.logical_not()).sum() / (obj_masks * obj_sem_masks.logical_not()).sum()
    obj_cls_post_loss_unmask = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks * obj_sem_masks).sum() / (obj_masks * obj_sem_masks).sum()
    total_loss = lm_cls_loss + match_loss + obj_cls_raw_loss + obj_cls_pre_loss_unmask + obj_cls_post_loss
    return total_loss, lm_cls_loss, match_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, obj_cls_pre_loss_mask, obj_cls_pre_loss_unmask, obj_cls_post_loss_mask, obj_cls_post_loss_unmask

@registry.register_optimizer("sqa_loss_v1")
def get_qa_loss_v1(obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, answer_scores, obj_labels, obj_masks, answer_label):
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    answer_loss = F.binary_cross_entropy_with_logits(answer_scores, answer_label.float(), reduction='sum') / answer_scores.shape[0]
    total_loss = obj_cls_raw_loss + obj_cls_pre_loss + obj_cls_post_loss + answer_loss
    return total_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, answer_loss


@registry.register_optimizer("caption_loss_v1")
def get_caption_loss_v1(txt_lm_cls_logits, masked_lm_labels, obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, obj_labels, obj_masks):
    loss_fct = CrossEntropyLoss(ignore_index=-1)
    lm_cls_loss = loss_fct(txt_lm_cls_logits.permute(0, 2, 1), masked_lm_labels)
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    total_loss = lm_cls_loss + obj_cls_raw_loss + obj_cls_pre_loss + obj_cls_post_loss
    return total_loss, lm_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss