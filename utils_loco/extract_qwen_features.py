import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import argparse
from tqdm import tqdm
from loader_loco_qwen import get_hd_data_loader
from config_loco import PROMPTS_QWEN
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from models.teacher_qwen import QwenViTFeatureExtractor, QwenLLMFeatureExtractor

def extract(args):
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    
    target_layers = args.layers
    
    if args.students_blocks == 'ViT':
        teacher = QwenViTFeatureExtractor(model, processor, layers=target_layers)
    else:
        teacher = QwenLLMFeatureExtractor(model, processor, layers=target_layers)
        prompt_key = f'assistant_generic'   ## CHANGE PROMPT HERE
        if prompt_key not in PROMPTS_QWEN:
            raise ValueError(f"Prompt {prompt_key} missing in PROMPTS_CONFIG")
        else:
            conv_template = PROMPTS_QWEN[prompt_key]

    for split in ['train', 'validation']:
        loader = get_hd_data_loader(split, args.class_name, args.dataset_path, batch_size=1)
        save_base = f"./features/{args.class_name}/{args.students_blocks}/{split}"
        os.makedirs(save_base, exist_ok=True)
        
        for pil_images, filenames in tqdm(loader, desc=f"Extracting {split}"):
            with torch.no_grad():
                f1, f2 = teacher(pil_images) if args.students_blocks == 'ViT' else teacher(pil_images, conv_template)
                
                torch.save(f1[0].cpu(), os.path.join(save_base, f"{filenames[0]}_layer_{target_layers[0]}.pt"))
                torch.save(f2[0].cpu(), os.path.join(save_base, f"{filenames[0]}_layer_{target_layers[1]}.pt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extraction Framework')

    parser.add_argument('--dataset_path', default = './datasets/mvtec_loco', type = str, 
                        help = 'Dataset path.')
    
    parser.add_argument('--class_name', default = 'breakfast_box', type = str,
                        help = 'Category name.')
    
    parser.add_argument('--students_blocks', type=str, default='ViT', choices=['ViT', 'LLM'],
                        help='Training scenario: where the 2 students extract features from')
    
    parser.add_argument('--layers', type=int, nargs=2, default=[11, 13],
                        help='2 layers to be extracted')

    args = parser.parse_args()
    extract(args)