import argparse
import os
import sys
import torch
import wandb
from itertools import chain
from tqdm import tqdm, trange

from utils_loco.loader_loco import get_data_loader 
from utils_loco.general_utils import set_seeds, SinusoidalPositionalEmbedding2D
from utils_loco.config_loco import PROMPTS_CONFIG

from models.teacher import ViTFeatureExtractor, LLMFeatureExtractor
from models.student import ResidualFeatureProjectionMLP, FeatureProjectionMLP, FeatureProjectionBottleneckMLP

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(args):
    set_seeds()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    model_name = f'{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'

    wandb.init(
        project='L2BT_MLLM',
        name=model_name,
        # mode="disabled"
    )

    # Dataloaders.
    train_loader, _ = get_data_loader(
        "train", class_name=args.class_name,
        img_size=args.img_size, batch_size=args.batch_size,
        dataset_path=args.dataset_path
    )
    
    val_loader, _ = get_data_loader(
        "validation", class_name=args.class_name,
        img_size=args.img_size, batch_size=args.batch_size,
        dataset_path=args.dataset_path
    )

    # Positional encoding dimension to be concatenated
    pe_dim = args.pe_dim

    # Model Initialization.
    students = {}
    teacher = None

    if args.students_blocks == 'Both_ViT':
        # --- ViT Bidirectional, Standard MLP ---
        
        # Teacher
        teacher = ViTFeatureExtractor(layers=args.layers).to(device, dtype=dtype).eval()
        
        # Students
        students['backward_net'] = FeatureProjectionMLP(
            in_features=teacher.embed_dim, out_features=teacher.embed_dim, pe_dim=pe_dim, reduction_factor=0.4
        ).to(device, dtype=dtype)
        students['forward_net'] = FeatureProjectionMLP(
            in_features=teacher.embed_dim, out_features=teacher.embed_dim, pe_dim=pe_dim, reduction_factor=0.4
        ).to(device, dtype=dtype)

        grid_size = args.img_size // teacher.patch_size

    elif args.students_blocks == 'Both_LLM':
        # --- LLM Bidirectional, Standard MLP ---

        # Teacher
        # if f'distilled_{args.class_name}' not in PROMPTS_CONFIG:
        #     raise ValueError(f"\nERROR: Prompt for '{args.class_name}' class is missing.\n")
        teacher = LLMFeatureExtractor(conversation_template=PROMPTS_CONFIG[f'generic'],
                                      layer1_idx=args.layers[0], layer2_idx=args.layers[1]).to(device).eval()
        
        # Students
        students['backward_net'] = FeatureProjectionMLP(
            in_features=teacher.embed_dim, out_features=teacher.embed_dim, pe_dim=pe_dim, act_layer=torch.nn.SiLU, reduction_factor=0.4
        ).to(device=device, dtype=dtype)
        students['forward_net'] = FeatureProjectionMLP(
            in_features=teacher.embed_dim, out_features=teacher.embed_dim, pe_dim=pe_dim, act_layer=torch.nn.SiLU, reduction_factor=0.4
        ).to(device=device, dtype=dtype)

        grid_size = 14

    else:
        raise ValueError("Invalid students_blocks option")
    
    # Positional Encoding generator
    pe_generator = SinusoidalPositionalEmbedding2D(pe_dim=pe_dim, size_h=grid_size).to(device, dtype=dtype)
    pe_alpha = args.pe_alpha

    # Optimizer
    optimizer = torch.optim.Adam(
        params=chain(students['backward_net'].parameters(), students['forward_net'].parameters()), 
        lr=1e-4
    )
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)

    # --- Training Loop ---
    for epoch in trange(args.epochs_no, desc=f'Training students for {args.class_name}...'):

        for model in students.values():
            model.train()
            
        global_loss = []

        for pil_img, tensor_img in tqdm(train_loader, desc=f'    Epoch {epoch+1} [Train]'):
            
            # Prepare Input
            if args.students_blocks == 'Both_ViT':
                input_data = tensor_img.to(device, dtype=dtype)
            else:
                input_data = pil_img

            # Feature extraction
            with torch.no_grad():
                earlier_patch, later_patch = teacher(input_data)

            # Positional Embeddings computation
            pe_matrix = pe_generator(earlier_patch)

            # if epoch == 0 and global_loss == []:
            #         with torch.no_grad():
            #             feat_norm = earlier_patch.norm(p=2, dim=-1).mean()
                        
            #             # Calcoliamo la norma del PE scalato
            #             scaled_pe = pe_matrix * pe_alpha 
            #             pe_norm = scaled_pe.norm(p=2, dim=-1).mean()
                        
            #             ratio = pe_norm / (feat_norm + 1e-6)

            #             print("\n" + "="*50)
            #             print(f"MAGNITUDE CHECK (Alpha={pe_alpha})")
            #             print(f"   -> Feature Mean Norm: {feat_norm.item():.4f}")
            #             print(f"   -> PE Mean Norm:      {pe_norm.item():.4f}")
            #             print(f"   -> Ratio (PE/Feat):   {ratio.item():.4f}")
                        
            #             if ratio > 0.35:
            #                 print("   TOO MUCH HIGH!")
            #                 sys.exit(1)
            #             else:
            #                 print("   GOOD BALANCE. Proceed...")
            #             print("="*50 + "\n")

            # Embedding Enrichment
            # earlier_enriched = earlier_patch + (pe_matrix * pe_alpha)
            # later_enriched = later_patch + (pe_matrix * pe_alpha)
            earlier_enriched = torch.cat([earlier_patch, pe_matrix * pe_alpha], dim=-1)
            later_enriched = torch.cat([later_patch, pe_matrix * pe_alpha], dim=-1)

            # Nets prediction
            predicted_later_patch = students['forward_net'](earlier_enriched)
            predicted_earlier_patch = students['backward_net'](later_enriched)

            # Losses
            loss_later = 1 - cos_sim(predicted_later_patch.float(), later_patch.float()).mean()
            loss_earlier = 1 - cos_sim(predicted_earlier_patch.float(), earlier_patch.float()).mean()
            loss = loss_later + loss_earlier

            # Logging & Optimization
            global_loss.append(loss.item())
            wandb.log({"train/batch_loss": loss.item()})

            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                print("Loss is NaN/Inf. Exiting.")
                sys.exit(1)

        epoch_train_loss = torch.tensor(global_loss).mean()
        wandb.log({"train/epoch_loss": epoch_train_loss.item()})


        # --- Validation ---
        for model in students.values():
            model.eval()
            
        global_val_loss = []

        with torch.no_grad():
            for pil_img, tensor_img in tqdm(val_loader, desc=f'    Epoch {epoch+1} [Val]'):
                
                # Prepare Input
                if args.students_blocks == 'Both_ViT':
                    input_data = tensor_img.to(device, dtype=dtype)
                else:
                    input_data = pil_img

                earlier_patch, later_patch = teacher(input_data)

                pe_matrix = pe_generator(earlier_patch)

                earlier_enriched = torch.cat([earlier_patch, pe_matrix * pe_alpha], dim=-1)
                later_enriched = torch.cat([later_patch, pe_matrix * pe_alpha], dim=-1)

                predicted_later_patch = students['forward_net'](earlier_enriched)
                predicted_earlier_patch = students['backward_net'](later_enriched)

                loss_later = 1 - cos_sim(predicted_later_patch.float(), later_patch.float()).mean()
                loss_earlier = 1 - cos_sim(predicted_earlier_patch.float(), earlier_patch.float()).mean()
                loss = loss_later + loss_earlier

                global_val_loss.append(loss.item())

        epoch_val_loss = torch.tensor(global_val_loss).mean()
        wandb.log({"val/epoch_loss": epoch_val_loss.item()})

        print(f"Epoch {epoch+1}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")

    # --- Model Saving ---
    directory = f'{args.checkpoint_savepath}/{args.class_name}'
    os.makedirs(directory, exist_ok=True)

    torch.save(students['forward_net'].state_dict(), os.path.join(directory, 'forward_net_' + model_name + '.pth'))
    torch.save(students['backward_net'].state_dict(), os.path.join(directory, 'backward_net_' + model_name + '.pth'))

    wandb.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Training Framework')

    parser.add_argument('--dataset_path', default = './datasets/mvtec_loco', type = str, 
                        help = 'Dataset path.')
    
    parser.add_argument('--checkpoint_savepath', default = './checkpoints/checkpoints_loco', type = str, 
                        help = 'Where to save the model checkpoints.')
    
    parser.add_argument('--class_name', default = 'breakfast_box', type = str,
                        help = 'Category name.')
    
    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs to train.')
    
    parser.add_argument('--img_size', default = 384, type = int,
                        help = 'Square image resolution.')
    
    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')
    
    parser.add_argument('--students_blocks', type=str, default='Both_LLM', choices=['Both_ViT', 'Both_LLM'],
                        help='Training scenario: where the 2 students extract features from')
    
    parser.add_argument('--label', default='ds_vit_pe010_11_15', type=str,
                        help='Label to identify the experiment.')
    
    parser.add_argument('--layers', type=int, nargs=2, default=[0, 1],
                        help='2 layers to be extracted')
    
    parser.add_argument('--pe_alpha', type=float, default=0.1,
                        help='Scaling factor for Positional Encoding')
    
    parser.add_argument('--pe_dim', type=int, default=128,
                        help='Positional encoding dimension to be concatenated')

    args = parser.parse_args()
    train(args)