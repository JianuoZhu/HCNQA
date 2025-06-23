import torch
import torch.nn as nn

import model.vision.pointnet2.pointnet2_utils as pointnet2_utils
from model.vision.basic_modules import get_mlp_head
from pipeline.registry import registry

@registry.register_other_model("supervise_base_head_v1")
class SuperviseBaseHeadV3(nn.Module):
    def __init__(self, input_size=768, hidden_size=768, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, obj_embeds):
        x = self.fc1(obj_embeds)
        x = self.fc2(x)
        return x
    
if __name__ == '__main__':
    SuperviseBaseHeadV3()
