import torch
from torchsummary import summary

class FeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features = None, out_features = None, pe_dim=0, act_layer = torch.nn.GELU, reduction_factor = 1):  # GELU for ViT student and SiLU for LLM student
        super().__init__()

        total_input = in_features + pe_dim
        hidden_features = int((total_input + out_features) // (2 * reduction_factor))

        self.input = torch.nn.Linear(total_input, hidden_features)
        self.projection = torch.nn.Linear(hidden_features, hidden_features)
        self.output = torch.nn.Linear(hidden_features, out_features)

        self.act_fcn = act_layer()

    def forward(self, x):
        x = self.input(x)
        x = self.act_fcn(x)

        x = self.projection(x)
        x = self.act_fcn(x)

        x = self.output(x)

        return x
    


class ResidualFeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features, out_features, act_layer=torch.nn.GELU, reduction_factor = 1):
        super().__init__()
        self.act_fcn = act_layer()

        hidden_features = int((in_features + out_features) // (2 * reduction_factor))

        self.residual_block = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.LayerNorm(hidden_features),
            act_layer(),
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.LayerNorm(hidden_features),
            act_layer(),
            torch.nn.Linear(hidden_features, out_features),
            #act_layer(),
        )

        if in_features != out_features:
            self.shortcut = torch.nn.Linear(in_features, out_features)
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        shortcut_x = self.shortcut(x)

        residual = self.residual_block(x)

        return shortcut_x + residual




class NormalizedFeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features, out_features, act_layer=torch.nn.GELU, reduction_factor=1, use_norm=True):
        super().__init__()

        self.act_fcn = act_layer()
        self.use_norm = use_norm

        hidden_features = int((in_features + out_features) / (2 * reduction_factor))

        self.input = torch.nn.Linear(in_features, hidden_features)
        self.projection = torch.nn.Linear(hidden_features, hidden_features)
        self.output = torch.nn.Linear(hidden_features, out_features)

        if self.use_norm:
            self.norm1 = torch.nn.LayerNorm(hidden_features)
            self.norm2 = torch.nn.LayerNorm(hidden_features)

    def forward(self, x):
        x = self.input(x)
        if self.use_norm:
            x = self.norm1(x)
        x = self.act_fcn(x)

        x = self.projection(x)
        if self.use_norm:
            x = self.norm2(x)
        x = self.act_fcn(x)

        x = self.output(x)

        return x



class FeatureProjectionBottleneckMLP(torch.nn.Module):
    def __init__(self, in_features, out_features, reduction_factor = 1, act_layer=torch.nn.GELU):
        super().__init__()

        self.act_fcn = act_layer()

        intermediate1_dim = int((in_features + out_features) // (2 * reduction_factor))
        intermediate2_dim = int((in_features + out_features) // (3 * reduction_factor))
        bottleneck_dim = int((in_features + out_features) // (4 * reduction_factor))

        self.input = torch.nn.Linear(in_features, intermediate1_dim)
        self.encoder = torch.nn.Linear(intermediate1_dim, intermediate2_dim)
        self.bottleneck = torch.nn.Linear(intermediate2_dim, bottleneck_dim)
        self.decoder1 = torch.nn.Linear(bottleneck_dim, intermediate2_dim)
        self.decoder2 = torch.nn.Linear(intermediate2_dim, intermediate1_dim)
        self.output = torch.nn.Linear(intermediate1_dim, out_features)

    def forward(self, x):
        x = self.input(x)
        x = self.act_fcn(x)

        x = self.encoder(x)
        x = self.act_fcn(x)

        x = self.bottleneck(x)
        x = self.act_fcn(x)

        x = self.decoder1(x)
        x = self.act_fcn(x)

        x = self.decoder2(x)
        x = self.act_fcn(x)

        x = self.output(x)
        
        return x



class RegularizedFeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features, out_features, act_layer=torch.nn.GELU, reduction_factor=1, dropout_prob=0.3):
        super().__init__()
        self.act_fcn = act_layer()

        hidden_features = int((in_features + out_features) / (2 * reduction_factor))

        self.input = torch.nn.Linear(in_features, hidden_features)
        self.projection = torch.nn.Linear(hidden_features, hidden_features)
        self.output = torch.nn.Linear(hidden_features, out_features)

        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.input(x)
        x = self.act_fcn(x)
        x = self.dropout(x)

        x = self.projection(x)
        x = self.act_fcn(x)
        x = self.dropout(x)

        x = self.output(x)

        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        From https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2/modeling_qwen2.py
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class QwenAlignedStudent(torch.nn.Module):
    def __init__(self, in_features, out_features, act_layer=torch.nn.SiLU, reduction_factor=1):
        super().__init__()
        hidden_features = int((in_features + out_features) // (2 * reduction_factor))

        self.norm_in = RMSNorm(in_features, eps=1e-6)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            act_layer(),
            torch.nn.Linear(hidden_features, hidden_features),
            act_layer(),
            torch.nn.Linear(hidden_features, out_features),
        )
        
        if in_features != out_features:
            self.shortcut = torch.nn.Linear(in_features, out_features)
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.norm_in(x)
        x = self.mlp(x)
        return residual + x


class QwenViTAlignedStudent(torch.nn.Module):
    def __init__(self, in_features, out_features, act_layer=torch.nn.GELU, reduction_factor=1):
        super().__init__()
        hidden_features = int((in_features + out_features) // (2 * reduction_factor))

        self.norm_in = torch.nn.LayerNorm(in_features, eps=1e-6)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            act_layer(),
            torch.nn.Linear(hidden_features, hidden_features),
            act_layer(),
            torch.nn.Linear(hidden_features, out_features),
        )
        
        if in_features != out_features:
            self.shortcut = torch.nn.Linear(in_features, out_features)
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.norm_in(x)
        x = self.mlp(x)
        return residual + x
    

class FeatureProjectionMLP_TrainableScale(torch.nn.Module):
    def __init__(self, in_features=None, out_features=None, pe_dim=0, act_layer=torch.nn.GELU, reduction_factor=1):
        super().__init__()
        self.pe_alpha = torch.nn.Parameter(torch.tensor(0.75))
        
        total_input = in_features + pe_dim
        hidden_features = int((total_input + out_features) // (2 * reduction_factor))

        self.input = torch.nn.Linear(total_input, hidden_features)
        self.projection = torch.nn.Linear(hidden_features, hidden_features)
        self.output = torch.nn.Linear(hidden_features, out_features)
        self.act_fcn = act_layer()

    def forward(self, x, pe_matrix):
        x_enriched = torch.cat([x, pe_matrix * self.pe_alpha], dim=-1)
        
        x = self.act_fcn(self.input(x_enriched))
        x = self.act_fcn(self.projection(x))
        return self.output(x)
    

class FeatureProjectionMLP_FullyTrainable(torch.nn.Module):
    def __init__(self, in_features=None, out_features=None, pe_dim=128, grid_size=14, act_layer=torch.nn.GELU, reduction_factor=1):
        super().__init__()
        self.trainable_pe = torch.nn.Parameter(torch.randn(1, grid_size**2, pe_dim) * 0.02)
        
        total_input = in_features + pe_dim
        hidden_features = int((total_input + out_features) // (2 * reduction_factor))

        self.input = torch.nn.Linear(total_input, hidden_features)
        self.projection = torch.nn.Linear(hidden_features, hidden_features)
        self.output = torch.nn.Linear(hidden_features, out_features)
        self.act_fcn = act_layer()

    def forward(self, x):
        pe = self.trainable_pe.expand(x.size(0), -1, -1)
        x_enriched = torch.cat([x, pe], dim=-1)
        
        x = self.act_fcn(self.input(x_enriched))
        x = self.act_fcn(self.projection(x))
        return self.output(x)
    
class FeatureProjectionMLP_TrainableProj(torch.nn.Module):
    def __init__(self, in_features=None, out_features=None, pe_dim=128, act_layer=torch.nn.GELU, reduction_factor=1):
        super().__init__()
        self.pe_mapper = torch.nn.Linear(pe_dim, pe_dim)
        
        total_input = in_features + pe_dim
        hidden_features = int((total_input + out_features) // (2 * reduction_factor))

        self.input = torch.nn.Linear(total_input, hidden_features)
        self.projection = torch.nn.Linear(hidden_features, hidden_features)
        self.output = torch.nn.Linear(hidden_features, out_features)
        self.act_fcn = act_layer()

    def forward(self, x, pe_matrix):

        mapped_pe = self.pe_mapper(pe_matrix)
        x_enriched = torch.cat([x, mapped_pe], dim=-1)
        
        x = self.act_fcn(self.input(x_enriched))
        x = self.act_fcn(self.projection(x))
        return self.output(x)
    
import torch
import math

class FeatureProjectionMLP_TrainableScaleSum(torch.nn.Module):
    def __init__(self, in_features=None, out_features=None, pe_dim=0, act_layer=torch.nn.GELU, reduction_factor=1):
        super().__init__()

        self.pe_alpha = torch.nn.Parameter(torch.tensor(0.05))
        
        total_input = in_features 
        
        hidden_features = int((total_input + out_features) // (2 * reduction_factor))

        self.input = torch.nn.Linear(total_input, hidden_features)
        self.projection = torch.nn.Linear(hidden_features, hidden_features)
        self.output = torch.nn.Linear(hidden_features, out_features)
        self.act_fcn = act_layer()

    def forward(self, x, pe_matrix):
        x_enriched = x + (pe_matrix * self.pe_alpha)
        
        x = self.act_fcn(self.input(x_enriched))
        x = self.act_fcn(self.projection(x))
        return self.output(x)


class FeatureProjectionMLP_FullyTrainableSum(torch.nn.Module):
    def __init__(self, in_features, out_features, grid_size=14, act_layer=torch.nn.GELU, reduction_factor=0.4):
        super().__init__()

        num_patches = grid_size**2
        
        self.trainable_pe = torch.nn.Parameter(torch.randn(1, num_patches, in_features) * 0.02)
        
        hidden_features = int((in_features + out_features) // (2 * reduction_factor))

        self.input = torch.nn.Linear(in_features, hidden_features)
        self.projection = torch.nn.Linear(hidden_features, hidden_features)
        self.output = torch.nn.Linear(hidden_features, out_features)
        self.act_fcn = act_layer()

    def forward(self, x):
        
        x_enriched = x + self.trainable_pe
        
        x = self.act_fcn(self.input(x_enriched))
        x = self.act_fcn(self.projection(x))
        return self.output(x)
    

class FeatureProjectionMLP_FullyTrainableSumHD(torch.nn.Module):
    def __init__(self, in_features, out_features, num_patches, act_layer=torch.nn.GELU, reduction_factor=0.4):
        super().__init__()

        self.num_patches = num_patches
        
        self.trainable_pe = torch.nn.Parameter(torch.randn(1, num_patches, in_features) * 0.02)
        
        hidden_features = int((in_features + out_features) // (2 * reduction_factor))
        self.input = torch.nn.Linear(in_features, hidden_features)
        self.projection = torch.nn.Linear(hidden_features, hidden_features)
        self.output = torch.nn.Linear(hidden_features, out_features)
        self.act_fcn = act_layer()

    def forward(self, x):
        x_enriched = x + self.trainable_pe
        x = self.act_fcn(self.input(x_enriched))
        x = self.act_fcn(self.projection(x))
        return self.output(x)



if __name__ == "__main__":
    IN_FEATURES = OUT_FEATURES = 3584
    REDUCTION = 0.4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_model = QwenAlignedStudent(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        reduction_factor=REDUCTION
    ).to(device)

    summary(student_model, input_size=(IN_FEATURES,))
