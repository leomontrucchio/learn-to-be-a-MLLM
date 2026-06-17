import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import torch.nn.functional as F
import wandb
from itertools import chain
from tqdm import tqdm, trange

from utils_loco.loader_loco_qwen import get_hd_data_loader 
from utils_loco.general_utils import set_seeds
from utils_loco.config_loco import PROMPTS_QWEN

from models.student import FeatureProjectionMLP, QwenAlignedStudent, QwenViTAlignedStudent


def train(args):
    set_seeds()
    device = "cuda"
    dtype = torch.bfloat16

    exp_name = f'{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'
    wandb.init(project='L2BT_MLLM_Qwen', name=exp_name)

    mode = args.students_blocks.split('_')[1]
    feature_dir = f"./features/{args.class_name}/{mode}"

    # DataLoader
    train_loader = get_hd_data_loader("train", args.class_name, args.dataset_path, args.batch_size,
                                      feature_dir=feature_dir, target_layers=args.layers)
    val_loader = get_hd_data_loader("validation", args.class_name, args.dataset_path, args.batch_size,
                                    feature_dir=feature_dir, target_layers=args.layers)
    
    # Students Initialization
    if args.students_blocks == 'Both_ViT':
        embed_dim = 1280
        students = {
            'forward_net': QwenViTAlignedStudent(embed_dim, embed_dim, reduction_factor=0.75).to(device, dtype=dtype),
            'backward_net': QwenViTAlignedStudent(embed_dim, embed_dim, reduction_factor=0.75).to(device, dtype=dtype)
            }
    else:
        embed_dim = 3584
        students = {
            'forward_net': QwenAlignedStudent(embed_dim, embed_dim, torch.nn.SiLU, reduction_factor=0.75).to(device, dtype=dtype),
            'backward_net': QwenAlignedStudent(embed_dim, embed_dim, torch.nn.SiLU, reduction_factor=0.75).to(device, dtype=dtype)
            }

    # Optimizer
    optimizer = torch.optim.Adam(params=chain(students['backward_net'].parameters(), students['forward_net'].parameters()), lr=1e-4)
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)
    
    # --- Training Loop ---
    for epoch in trange(args.epochs_no, desc=f'Training Qwen Students ({args.class_name})'):
        for s in students.values():
            s.train()
        global_loss = []

        for earlier_batch, later_batch in tqdm(train_loader):

            earlier_input = earlier_batch.to(device, dtype).view(-1, embed_dim)
            later_input = later_batch.to(device, dtype).view(-1, embed_dim)

            # Nets prediction
            pred_later = students['forward_net'](earlier_input)
            pred_earlier = students['backward_net'](later_input)

            pred_later_norm = F.normalize(pred_later.float(), p=2, dim=-1)
            later_input_norm = F.normalize(later_input.float(), p=2, dim=-1)
            pred_earlier_norm = F.normalize(pred_earlier.float(), p=2, dim=-1)
            earlier_input_norm = F.normalize(earlier_input.float(), p=2, dim=-1)

            # Losses
            loss_later = 1 - cos_sim(pred_later_norm, later_input_norm).mean()
            loss_earlier = 1 - cos_sim(pred_earlier_norm, earlier_input_norm).mean()
            loss = loss_later + loss_earlier

            # Logging and Optimization
            global_loss.append(loss.item())
            wandb.log({"train/batch_loss": loss.item()})

            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                print("Loss is NaN/Inf. Exiting.")
                exit()

        epoch_train_loss = torch.tensor(global_loss).mean()
        wandb.log({"train/epoch_loss": epoch_train_loss.item()})

        # --- Validation ---
        for s in students.values():
            s.eval()
        global_val_loss = []
        with torch.no_grad():
            for earlier_batch, later_batch in tqdm(val_loader, desc=f'    Epoch {epoch+1} [Val]'):

                earlier_input = earlier_batch.to(device, dtype).view(-1, embed_dim)
                later_input = later_batch.to(device, dtype).view(-1, embed_dim)

                predicted_later_patch = students['forward_net'](earlier_input)
                predicted_earlier_patch = students['backward_net'](later_input)

                p_later_norm = F.normalize(predicted_later_patch.float(), p=2, dim=-1)
                l_input_norm = F.normalize(later_input.float(), p=2, dim=-1)
                p_earlier_norm = F.normalize(predicted_earlier_patch.float(), p=2, dim=-1)
                e_input_norm = F.normalize(earlier_input.float(), p=2, dim=-1)
                
                loss_later = 1 - cos_sim(p_later_norm, l_input_norm).mean()
                loss_earlier = 1 - cos_sim(p_earlier_norm, e_input_norm).mean()
                loss = loss_later + loss_earlier

                global_val_loss.append(loss.item())
        
        epoch_val_loss = torch.tensor(global_val_loss).mean()
        wandb.log({"val/epoch_loss": epoch_val_loss.item()})

        print(f"Epoch {epoch+1}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")

    # --- Model Saving ---
    save_dir = f'{args.checkpoint_savepath}/{args.class_name}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(students['forward_net'].state_dict(), os.path.join(save_dir, f'forward_net_{exp_name}.pth'))
    torch.save(students['backward_net'].state_dict(), os.path.join(save_dir, f'backward_net_{exp_name}.pth'))
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
    
    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension.')
    
    parser.add_argument('--students_blocks', type=str, default='Both_ViT', choices=['Both_ViT', 'Both_LLM'],
                        help='Training scenario: where the 2 students extract features from')
    
    parser.add_argument('--label', default='qwen_vit_11_13', type=str,
                        help='Label to identify the experiment.')
    
    parser.add_argument('--layers', type=int, nargs=2, default=[11, 13],
                        help='2 layers to be extracted')

    args = parser.parse_args()
    train(args)