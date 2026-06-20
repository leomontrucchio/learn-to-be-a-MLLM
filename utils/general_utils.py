import torch
import math
import numpy as np
from torchvision import transforms

def set_seeds(sid=115):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)

# Helper function to extract metrics safely (for LOCO)
def extract_metrics_loco(category_key, metrics):
    i_auroc = metrics['classification']['auc_roc'].get(category_key, 0.0)
    spro_dict = metrics['localization']['auc_spro'].get(category_key, {})
    spro_30 = spro_dict.get('0.3', 0.0)
    spro_10 = spro_dict.get('0.1', 0.0)
    spro_05 = spro_dict.get('0.05', 0.0)
    spro_01 = spro_dict.get('0.01', 0.0)
    return i_auroc, spro_30, spro_10, spro_05, spro_01

siglip_denormalize = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5]),
    transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.])
])

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.functional.pad(image, padding, padding_mode='edge')

class SinusoidalPositionalEmbedding2D(torch.nn.Module):
    """
    Generate a fixed 2D Sinusoidal Positional Encoding 2D.
    """
    def __init__(self, pe_dim, size_h): 
        super().__init__()
        self.pe_dim = pe_dim
        self.size_h = size_h

        pe = self._make_grid_pe(size_h, pe_dim)
        self.register_buffer("pe", pe)

    def _make_grid_pe(self, size, d_model):
        coords = torch.arange(size, dtype=torch.float32)
        
        d_half = d_model // 2
        div_term = torch.exp(torch.arange(0, d_half, 2).float() * -(math.log(10000.0) / d_half))
        
        pos_emb = torch.zeros(size, d_half)
        pos_emb[:, 0::2] = torch.sin(coords.unsqueeze(1) * div_term)
        pos_emb[:, 1::2] = torch.cos(coords.unsqueeze(1) * div_term)
        
        pe_y = pos_emb.unsqueeze(1).repeat(1, size, 1) # (H, W, D/2)
        pe_x = pos_emb.unsqueeze(0).repeat(size, 1, 1) # (H, W, D/2)
        
        pe = torch.cat([pe_y, pe_x], dim=-1)
        pe = pe.flatten(0, 1).unsqueeze(0) # (1, H*W, D)
        return pe

    def forward(self, x):
        pe_slice = self.pe[:, :x.size(1), :]
        return pe_slice.expand(x.size(0), -1, -1)
