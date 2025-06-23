import torch
import torch.nn as nn

import model.vision.pointnet2.pointnet2_utils as pointnet2_utils
from model.vision.basic_modules import get_mlp_head
from pipeline.registry import registry

@registry.register_other_model("coarse_ground_head_v4")
class CoarseGroundHeadV4(nn.Module):
    def __init__(self, input_size=768, hidden_size=768, dropout=0.3):
        super().__init__()

        self.feature_extract_1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(dropout)
        )

        self.feature_extract_2 = nn.Sequential(
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
        
    def forward(self, sup_embeds):
        out1 = self.feature_extract_1(sup_embeds)
        out2 = self.feature_extract_2(sup_embeds)
        out = torch.cat((out1, out2), dim=-1)
        out = self.fuse_block(out)
        cg_res = out
        out = self.output_layer(out)
        cg_logits = out.squeeze(2)
        cg_mask = self.sig(cg_logits)
        return cg_res, cg_logits, cg_mask
    
if __name__ == '__main__':
    CoarseGroundHeadV4()