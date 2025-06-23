# utils/analysis.py
import torch

def count_parameters(model: torch.nn.Module, only_trainable=False):
    """Return total or trainable parameter count."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def pretty_millions(n):
    return f'{n/1e6:,.2f} M'

def summarize_pipeline(pipeline):
    """Assumes the sub-modules are attributes on your Pipeline object."""
    submods = [
        ('lang_encoder', pipeline.lang_encoder),
        ('point_encoder', pipeline.point_encoder),
        ('unified_encoder', pipeline.unified_encoder),
        ('ground_head', pipeline.ground_head),
        ('qa_head', pipeline.qa_head),
        ('pretrain_head', pipeline.pretrain_head),
        ('caption_head', pipeline.caption_head),
        ('cg_head', pipeline.cghead),
        ('fg_head', pipeline.fghead),
        ('inference_head', pipeline.inference_head),
        ('sup_head', pipeline.suphead)
    ]

    print(f'{"Module":<18}{"Total":>12}{"Trainable":>15}')
    print('-' * 45)
    tot, train = 0, 0
    for name, m in submods:
        p_all  = count_parameters(m)
        p_trn  = count_parameters(m, True)
        print(f'{name:<18}{pretty_millions(p_all):>12}{pretty_millions(p_trn):>15}')
        tot  += p_all
        train += p_trn
    print('-' * 45)
    print(f'{"PIPELINE":<18}{pretty_millions(tot):>12}{pretty_millions(train):>15}')
    return tot, train
