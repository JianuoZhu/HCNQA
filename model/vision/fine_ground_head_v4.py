import torch
import torch.nn as nn

import model.vision.pointnet2.pointnet2_utils as pointnet2_utils
from model.vision.basic_modules import get_mlp_head
from pipeline.registry import registry

@registry.register_other_model("fine_ground_head_v4")
class FineGroundHeadV4(nn.Module):
    def __init__(self, input_size=768, hidden_size=768, dropout=0.3):
        super().__init__()

        self.feature_extract = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(dropout)
        )

        self.fuse_block = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(dropout)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size//2, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 1)
        )
        self.sig = nn.Sigmoid()
        self.adjust_dim = nn.Conv1d(in_channels=hidden_size * 2, out_channels=hidden_size//2, kernel_size=1)
        
    def forward(self, sup_embeds, cg_res):
        out1 = self.feature_extract(sup_embeds)
        out2 = cg_res
        out = torch.cat((out1, out2), dim=-1)
        out = self.fuse_block(out)
        fg_res = out
        out = self.output_layer(out)
        fg_logits = out.squeeze(2)
        fg_mask = self.sig(fg_logits)
        return fg_res, fg_logits, fg_mask
    
if __name__ == '__main__':
    FineGroundHeadV4()
