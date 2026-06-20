import argparse
import os
import sys
import torch
import wandb
from itertools import chain
from tqdm import tqdm, trange
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

from utils.data_loader import get_data_loader 
from utils.general_utils import set_seeds, SinusoidalPositionalEmbedding2D
from utils.prompts import get_prompt

from models.teacher import ViTFeatureExtractor, LLMFeatureExtractor, QwenViTFeatureExtractor, QwenLLMFeatureExtractor
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

    train_loader, _ = get_data_loader(
        args.dataset_name, "train", class_name=args.class_name,
        img_size=args.img_size, batch_size=args.batch_size,
        dataset_path=args.dataset_path
    )
    
    val_loader, _ = get_data_loader(
        args.dataset_name, "validation", class_name=args.class_name,
        img_size=args.img_size, batch_size=args.batch_size,
        dataset_path=args.dataset_path
    )

    pe_dim = args.pe_dim
    students = {}
    teachers = {}
    pe_generators = {}

    use_pe = (args.mllm == 'deepseek' and args.dataset_name == 'loco')
    actual_pe_dim = pe_dim if use_pe else 0
    act_layer = torch.nn.SiLU if args.students_blocks == 'Both_LLM' else torch.nn.GELU
    if args.mllm == 'qwen': act_layer = torch.nn.SiLU

    if args.mllm == 'qwen':
        qwen_model_name = "Qwen/Qwen2-VL-7B-Instruct"
        shared_model = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        shared_processor = Qwen2VLProcessor.from_pretrained(qwen_model_name)
        prompt_template = get_prompt(args.dataset_name, 'qwen', 'generic')
    else:
        prompt_template = get_prompt(args.dataset_name, 'deepseek', 'generic')

    if args.students_blocks == 'Both_ViT':
        if args.mllm == 'deepseek':
            teachers['fe'] = ViTFeatureExtractor(layers=args.layers).to(device, dtype=dtype).eval()
            grid_size = args.img_size // teachers['fe'].patch_size
            embed_dim = teachers['fe'].embed_dim
        else:
            teachers['fe'] = QwenViTFeatureExtractor(model=shared_model, processor=shared_processor, layers=args.layers).to(device, dtype=dtype).eval()
            FACTOR = 28
            new_h, new_w = round(args.img_size / FACTOR) * FACTOR, round(args.img_size / FACTOR) * FACTOR
            grid_size = new_h // 14
            embed_dim = shared_model.visual.embed_dim

        if use_pe:
            pe_generators['vit'] = SinusoidalPositionalEmbedding2D(pe_dim=pe_dim, size_h=grid_size).to(device, dtype=dtype)
        
        students['backward_net'] = FeatureProjectionMLP(in_features=embed_dim, out_features=embed_dim, pe_dim=actual_pe_dim, act_layer=act_layer, reduction_factor=0.4).to(device=device, dtype=dtype)
        students['forward_net'] = FeatureProjectionMLP(in_features=embed_dim, out_features=embed_dim, pe_dim=actual_pe_dim, act_layer=act_layer, reduction_factor=0.4).to(device=device, dtype=dtype)

    elif args.students_blocks == 'Both_LLM':
        if args.mllm == 'deepseek':
            teachers['fe'] = LLMFeatureExtractor(conversation_template=prompt_template, layer1_idx=args.layers[0], layer2_idx=args.layers[1]).to(device).eval()
            grid_size = 14
            embed_dim = teachers['fe'].embed_dim
        else:
            teachers['fe'] = QwenLLMFeatureExtractor(model=shared_model, processor=shared_processor, layers=args.layers).to(device).eval()
            FACTOR = 28
            new_h, new_w = round(args.img_size / FACTOR) * FACTOR, round(args.img_size / FACTOR) * FACTOR
            grid_size = new_h // 28
            embed_dim = shared_model.config.hidden_size

        if use_pe:
            pe_generators['llm'] = SinusoidalPositionalEmbedding2D(pe_dim=pe_dim, size_h=grid_size).to(device, dtype=dtype)
        
        students['backward_net'] = FeatureProjectionMLP(in_features=embed_dim, out_features=embed_dim, pe_dim=actual_pe_dim, act_layer=act_layer, reduction_factor=0.4).to(device=device, dtype=dtype)
        students['forward_net'] = FeatureProjectionMLP(in_features=embed_dim, out_features=embed_dim, pe_dim=actual_pe_dim, act_layer=act_layer, reduction_factor=0.4).to(device=device, dtype=dtype)


    pe_alpha = args.pe_alpha
    optimizer = torch.optim.Adam(params=chain(*(m.parameters() for m in students.values())), lr=1e-4)
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)

    # --- Training Loop ---
    for epoch in trange(args.epochs_no, desc=f'Training students for {args.class_name}...'):
        for model in students.values():
            model.train()
        global_loss = []

        for pil_img, tensor_img in tqdm(train_loader, desc=f'    Epoch {epoch+1} [Train]'):

            # Prepare input
            if args.students_blocks == 'Both_ViT':
                input_data = pil_img if args.mllm == 'qwen' else tensor_img.to(device, dtype=dtype)
            else:
                input_data = pil_img

            # Feature extraction
            with torch.no_grad():
                if args.mllm == 'qwen' and args.students_blocks == 'Both_LLM':
                    earlier_patch, later_patch = teachers['fe'](input_data, prompt_template)
                else:
                    earlier_patch, later_patch = teachers['fe'](input_data)

            # Positional encoding enrichment
            if use_pe:
                pe_key = 'vit' if args.students_blocks == 'Both_ViT' else 'llm'
                pe_matrix = pe_generators[pe_key](earlier_patch)
                earlier_enriched = torch.cat([earlier_patch, pe_matrix * pe_alpha], dim=-1)
                later_enriched = torch.cat([later_patch, pe_matrix * pe_alpha], dim=-1)
            else:
                earlier_enriched = earlier_patch
                later_enriched = later_patch

            # Student predictions
            predicted_later_patch = students['forward_net'](earlier_enriched)
            predicted_earlier_patch = students['backward_net'](later_enriched)

            # Losses
            loss_later = 1 - cos_sim(predicted_later_patch.float(), later_patch.float()).mean()
            loss_earlier = 1 - cos_sim(predicted_earlier_patch.float(), earlier_patch.float()).mean()
            loss = loss_later + loss_earlier

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
                if args.students_blocks == 'Both_ViT':
                    input_data = tensor_img.to(device, dtype=dtype)
                    if args.mllm == 'qwen':
                        input_data = pil_img
                else:
                    input_data = pil_img

                if args.mllm == 'qwen' and args.students_blocks == 'Both_LLM':
                    earlier_patch, later_patch = teachers['fe'](input_data, prompt_template)
                else:
                    earlier_patch, later_patch = teachers['fe'](input_data)

                if use_pe:
                    pe_key = 'vit' if args.students_blocks == 'Both_ViT' else 'llm'
                    pe_matrix = pe_generators[pe_key](earlier_patch)
                    earlier_enriched = torch.cat([earlier_patch, pe_matrix * pe_alpha], dim=-1)
                    later_enriched = torch.cat([later_patch, pe_matrix * pe_alpha], dim=-1)
                else:
                    earlier_enriched = earlier_patch
                    later_enriched = later_patch

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

    for name, model in students.items():
        save_name = f'{name}_{model_name}.pth'
        torch.save(model.state_dict(), os.path.join(directory, save_name))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Training Framework')

    parser.add_argument('--dataset_name', default='loco', type=str, choices=['loco', 'visa'],
                        help='Dataset name.')
    
    parser.add_argument('--dataset_path', default='./datasets/mvtec_loco', type=str, 
                        help='Dataset path.')
    
    parser.add_argument('--checkpoint_savepath', default='./checkpoints/checkpoints_loco', type=str, 
                        help='Where to save the model checkpoints.')
    
    parser.add_argument('--class_name', default='breakfast_box', type=str,
                        help='Category name.')
    
    parser.add_argument('--epochs_no', default=50, type=int,
                        help='Number of epochs to train.')
    
    parser.add_argument('--img_size', default=384, type=int,
                        help='Square image resolution.')
    
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch dimension.')
    
    parser.add_argument('--mllm', type=str, default='deepseek', choices=['deepseek', 'qwen'],
                        help='Which MLLM to use as teacher.')

    parser.add_argument('--students_blocks', type=str, default='Both_LLM', choices=['Both_ViT', 'Both_LLM'],
                        help='Training scenario: where the 2 students extract features from')
    
    parser.add_argument('--label', default='ds_llm_pe010', type=str,
                        help='Label to identify the experiment.')
    
    parser.add_argument('--layers', type=int, nargs=2, default=[0, 1],
                        help='2 layers to be extracted')
    
    parser.add_argument('--pe_alpha', type=float, default=0.1,
                        help='Scaling factor for Positional Encoding')
    
    parser.add_argument('--pe_dim', type=int, default=128,
                        help='Positional encoding dimension to be concatenated')

    args = parser.parse_args()
    train(args)
