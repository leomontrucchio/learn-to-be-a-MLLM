import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
import tifffile
import subprocess
import json
import glob
import matplotlib.pyplot as plt
from PIL import Image
import math

from utils_loco.hd_loader_loco import get_hd_data_loader
from utils_loco.general_utils import set_seeds, extract_metrics
from utils_loco.config_loco import PROMPTS_QWEN

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from models.teacher_qwen import QwenViTFeatureExtractor, QwenLLMFeatureExtractor
from models.student import FeatureProjectionMLP, QwenAlignedStudent, QwenViTAlignedStudent

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- Evaluation Pipeline ---
def run_evaluation_pipeline(args):
    eval_script_path = os.path.join("mvtec_loco_ad_evaluation", "evaluate_experiment.py")
    quantitative_dir = getattr(args, 'quantitative_folder', './results/loco/quantitatives_loco')
    os.makedirs(quantitative_dir, exist_ok=True)
    
    print(f"\nRunning evaluation for {args.class_name}...")
    
    cmd = [
        "python3", eval_script_path,
        "--object_name", args.class_name,
        "--dataset_base_dir", args.dataset_path,
        "--anomaly_maps_dir", args.anomaly_maps_dir,
        "--output_dir", quantitative_dir,
        "--num_parallel_workers", "12",
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation script: {e}")
        return

    metrics_path = os.path.join(quantitative_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"metrics.json not found at {metrics_path}")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    glob_auc, glob_30, glob_10, glob_05, glob_01 = extract_metrics('global', metrics)
    struct_auc, struct_30, struct_10, struct_05, struct_01 = extract_metrics('structural_anomalies', metrics)
    logic_auc, logic_30, logic_10, logic_05, logic_01 = extract_metrics('logical_anomalies', metrics)

    print(f"\nResults for {args.class_name}:")
    header = f"{'Type':<12} | {'I-AUROC':<8} | {'sPRO@30%':<8} | {'sPRO@10%':<8} | {'sPRO@5%':<8} | {'sPRO@1%':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(f"{'Global':<12} | {glob_auc:.3f}    | {glob_30:.3f}    | {glob_10:.3f}    | {glob_05:.3f}    | {glob_01:.3f}")
    print(f"{'Structural':<12} | {struct_auc:.3f}    | {struct_30:.3f}    | {struct_10:.3f}    | {struct_05:.3f}    | {struct_01:.3f}")
    print(f"{'Logical':<12} | {logic_auc:.3f}    | {logic_30:.3f}    | {logic_10:.3f}    | {logic_05:.3f}    | {logic_01:.3f}")
    print("-" * len(header))

    result_file_name = f'{quantitative_dir}/{args.students_blocks}_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.md'
    
    with open(result_file_name, "w") as f:
        f.write(f'Metrics for class {args.class_name}\n')
        f.write(f'Config: {args.epochs_no} epochs, {args.batch_size} batch size\n\n')
        
        f.write('| Type       | sPRO@30% | sPRO@10% | sPRO@5%  | sPRO@1%  | I-AUROC  |\n')
        f.write('| ---------- | -------- | -------- | -------- | -------- | -------- |\n')
        f.write(f'| Global     | {glob_30:.3f} | {glob_10:.3f} | {glob_05:.3f} | {glob_01:.3f} | {glob_auc:.3f} |\n')
        f.write(f'| Structural | {struct_30:.3f} | {struct_10:.3f} | {struct_05:.3f} | {struct_01:.3f} | {struct_auc:.3f} |\n')
        f.write(f'| Logical    | {logic_30:.3f} | {logic_10:.3f} | {logic_05:.3f} | {logic_01:.3f} | {logic_auc:.3f} |\n')

    results = {
        'class_name': [args.class_name],
        'global_i_auroc': [glob_auc],
        'global_spro_30': [glob_30], 'global_spro_10': [glob_10], 'global_spro_05': [glob_05], 'global_spro_01': [glob_01],
        'struct_i_auroc': [struct_auc],
        'struct_spro_30': [struct_30], 'struct_spro_10': [struct_10], 'struct_spro_05': [struct_05], 'struct_spro_01': [struct_01],
        'logic_i_auroc': [logic_auc],
        'logic_spro_30': [logic_30], 'logic_spro_10': [logic_10], 'logic_spro_05': [logic_05], 'logic_spro_01': [logic_01],
    }
    pd.DataFrame(results).to_csv(result_file_name.replace('md', 'csv'), index=False, sep=',')



# --- Inference function ---
def infer(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Dataloader
    test_loader = get_hd_data_loader("test", args.class_name, args.dataset_path, batch_size=1)

    # Loading Shared Teacher
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    shared_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, device_map="auto"
    ).eval()
    shared_processor = Qwen2VLProcessor.from_pretrained(model_id)

    # Model Initialization
    teachers = {}
    students = {}
    check_dir = f'{args.checkpoint_folder}/{args.class_name}'

    if args.students_blocks == 'Both_ViT':
        # --- ViT Bidirectional ---
        teachers['fe'] = QwenViTFeatureExtractor(shared_model, shared_processor, layers=args.vit_layers)
        dim = shared_model.visual.config.embed_dim
        students['fw'] = QwenViTAlignedStudent(dim, dim, reduction_factor=0.75).to(device, dtype)
        students['bw'] = QwenViTAlignedStudent(dim, dim, reduction_factor=0.75).to(device, dtype)

        fwd_path = os.path.join(check_dir, f'forward_net_{args.vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        bwd_path = os.path.join(check_dir, f'backward_net_{args.vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        
        students['fw'].load_state_dict(torch.load(fwd_path, map_location=device, weights_only=False))
        students['bw'].load_state_dict(torch.load(bwd_path, map_location=device, weights_only=False))

    elif args.students_blocks == 'Both_LLM':
        # --- LLM Bidirectional ---
        teachers['fe'] = QwenLLMFeatureExtractor(shared_model, shared_processor, layers=args.llm_layers)
        dim = shared_model.config.text_config.hidden_size
        students['fw'] = QwenAlignedStudent(dim, dim, torch.nn.SiLU, reduction_factor=0.75).to(device, dtype)
        students['bw'] = QwenAlignedStudent(dim, dim, torch.nn.SiLU, reduction_factor=0.75).to(device, dtype)

        fwd_path = os.path.join(check_dir, f'forward_net_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        bwd_path = os.path.join(check_dir, f'backward_net_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')

        students['fw'].load_state_dict(torch.load(fwd_path, map_location=device, weights_only=False))
        students['bw'].load_state_dict(torch.load(bwd_path, map_location=device, weights_only=False))

    elif args.students_blocks == 'Full':
        # --- ViT + LLM ---
        teachers['vit_fe'] = QwenViTFeatureExtractor(shared_model, shared_processor, layers=args.vit_layers)
        teachers['llm_fe'] = QwenLLMFeatureExtractor(shared_model, shared_processor, layers=args.llm_layers)
        
        v_dim = shared_model.visual.config.embed_dim
        l_dim = shared_model.config.text_config.hidden_size

        # ViT Students
        students['vit_fw'] = QwenViTAlignedStudent(v_dim, v_dim, torch.nn.GELU, 0.4).to(device, dtype)
        students['vit_bw'] = QwenViTAlignedStudent(v_dim, v_dim, torch.nn.GELU, 0.4).to(device, dtype)
        # LLM Students
        students['llm_fw'] = QwenAlignedStudent(l_dim, l_dim, torch.nn.SiLU, 1).to(device, dtype)
        students['llm_bw'] = QwenAlignedStudent(l_dim, l_dim, torch.nn.SiLU, 1).to(device, dtype)

        vit_fw_path = os.path.join(check_dir, f'forward_net_{args.vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        vit_bw_path = os.path.join(check_dir, f'backward_net_{args.vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        llm_fw_path = os.path.join(check_dir, f'forward_net_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        llm_bw_path = os.path.join(check_dir, f'backward_net_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')

        students['vit_fw'].load_state_dict(torch.load(vit_fw_path, map_location=device, weights_only=False))
        students['vit_bw'].load_state_dict(torch.load(vit_bw_path, map_location=device, weights_only=False))
        students['llm_fw'].load_state_dict(torch.load(llm_fw_path, map_location=device, weights_only=False))
        students['llm_bw'].load_state_dict(torch.load(llm_bw_path, map_location=device, weights_only=False))

    else:
        raise ValueError("Invalid students_blocks option.")
    
    for s in students.values():
        s.eval()

    # Gaussian Blur Setup
    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device=device)/(w_l**2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device)/(w_u**2)

    # Conversation template (just if Both_LLM or Full)
    conv_template = None
    if args.students_blocks in ['Both_LLM', 'Full']:
        prompt_key = f'assistant_generic'
        if prompt_key not in PROMPTS_QWEN:
            raise ValueError(f"Prompt {prompt_key} missing in PROMPTS_CONFIG")
        conv_template = PROMPTS_QWEN[prompt_key]

    base_output_dir = os.path.join(args.anomaly_maps_dir, args.class_name, 'test')

    # --- Inference Loop ---
    for pil_imgs, img_path_list in tqdm(test_loader, desc=f'Inference {args.class_name}'):
        # Get original dimensions
        orig_w, orig_h = pil_imgs[0].size

        with torch.no_grad():
            if args.students_blocks == 'Full':
                v_early, v_late = teachers['vit_fe'](pil_imgs)
                l_early, l_late = teachers['llm_fe'](pil_imgs, conv_template)

                # ViT Anomaly map
                v_fw_pred = F.normalize(students['vit_fw'](v_early.view(-1, v_early.shape[-1])).view(v_early.shape).float(), p=2, dim=-1)
                v_late_norm = F.normalize(v_late.float(), p=2, dim=-1)
                v_fw = 1 - F.cosine_similarity(v_fw_pred, v_late_norm, dim=-1)

                v_bw_pred = F.normalize(students['vit_bw'](v_late.view(-1, v_late.shape[-1])).view(v_late.shape).float(), p=2, dim=-1)
                v_early_norm = F.normalize(v_early.float(), p=2, dim=-1)
                v_bw = 1 - F.cosine_similarity(v_bw_pred, v_early_norm, dim=-1)
                v_map = (v_fw * v_bw).unsqueeze(1)

                # LLM Anomaly map
                l_fw_pred = F.normalize(students['llm_fw'](l_early.view(-1, l_early.shape[-1])).view(l_early.shape).float(), p=2, dim=-1)
                l_late_norm = F.normalize(l_late.float(), p=2, dim=-1)
                l_fw = 1 - F.cosine_similarity(l_fw_pred, l_late_norm, dim=-1)

                l_bw_pred = F.normalize(students['llm_bw'](l_late.view(-1, l_late.shape[-1])).view(l_late.shape).float(), p=2, dim=-1)
                l_early_norm = F.normalize(l_early.float(), p=2, dim=-1)
                l_bw = 1 - F.cosine_similarity(l_bw_pred, l_early_norm, dim=-1)
                l_map = (l_fw * l_bw).unsqueeze(1)

                l_map_up = torch.nn.functional.interpolate(l_map, size=(v_map.shape[-2:]), mode='bilinear')
                anomaly_map = v_map + l_map_up

            else:
                # Both_ViT or Both_LLM
                e, l = teachers['fe'](pil_imgs) if args.students_blocks == 'Both_ViT' else teachers['fe'](pil_imgs, conv_template)

                fw_pred = F.normalize(students['fw'](e.view(-1, e.shape[-1])).view(e.shape).float(), p=2, dim=-1)
                l_norm = F.normalize(l.float(), p=2, dim=-1)
                fw = 1 - F.cosine_similarity(fw_pred, l_norm, dim=-1)

                bw_pred = F.normalize(students['bw'](l.view(-1, l.shape[-1])).view(l.shape).float(), p=2, dim=-1)
                e_norm = F.normalize(e.float(), p=2, dim=-1)
                bw = 1 - F.cosine_similarity(bw_pred, e_norm, dim=-1)
                
                anomaly_map = (fw * bw).unsqueeze(1)

            # Upsample to Original Image
            anomaly_map = torch.nn.functional.interpolate(anomaly_map, size=(orig_h, orig_w), mode='bilinear')

            # Gaussian Blur
            for _ in range(5):
                anomaly_map = torch.nn.functional.conv2d(anomaly_map, padding=pad_l, weight=weight_l)
            for _ in range(3):
                anomaly_map = torch.nn.functional.conv2d(anomaly_map, padding=pad_u, weight=weight_u)

            # Save
            img_path = img_path_list[0]
            parent_dir = os.path.basename(os.path.dirname(img_path)) # ex. logical_anomalies
            file_name = os.path.basename(img_path) # ex. 000.png
            file_name_no_ext = os.path.splitext(file_name)[0] # ex. 000

            save_dir = os.path.join(base_output_dir, parent_dir)
            save_path = os.path.join(save_dir, f"{file_name_no_ext}.tiff")

            os.makedirs(save_dir, exist_ok=True)

            map_numpy = anomaly_map.squeeze().cpu().detach().numpy()
            tifffile.imwrite(save_path, map_numpy)

            # Qualitative Plot
            if args.produce_qualitatives:
                save_qual_path = f'{args.qualitative_folder}/{args.students_blocks}_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs/{parent_dir}'
                os.makedirs(save_qual_path, exist_ok=True)

                if parent_dir == 'good':
                    gt_np = np.zeros((orig_h, orig_w))
                else:
                    gt_folder_path = os.path.join(args.dataset_path, args.class_name, 'ground_truth', parent_dir, file_name_no_ext)
                    gt_files = glob.glob(os.path.join(gt_folder_path, "*.png"))
                    if not gt_files:
                        gt_np = np.zeros((orig_h, orig_w))
                    else:
                        gt_combined = None
                        for f in gt_files:
                            mask = np.array(Image.open(f).convert('L'))
                            mask = (mask > 0).astype(np.float32)
                            if gt_combined is None: gt_combined = mask
                            else: gt_combined = np.maximum(gt_combined, mask)
                        gt_np = gt_combined

                _, axs = plt.subplots(1, 3, figsize=(7, 3))
                
                # Visualize Input Image
                axs[0].imshow(pil_imgs[0])
                axs[0].set_title('Input Image')
                axs[1].imshow(gt_np, cmap='gray')
                axs[1].set_title('Ground-truth')
                axs[2].imshow(map_numpy, cmap=plt.cm.jet)
                axs[2].set_title('Anomaly Map')

                for ax in axs.flat: ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_qual_path, file_name_no_ext), dpi=256)
                plt.close()

    print("Inference complete. Maps are stored.")
    run_evaluation_pipeline(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Inference Framework')

    parser.add_argument('--checkpoint_folder', default = './checkpoints/checkpoints_loco', type = str,
                        help = 'Path to the folder containing students checkpoints.')
    
    parser.add_argument('--anomaly_maps_dir', default='./results/loco/anomaly_maps', type=str,
                        help='Base directory where anomaly maps will be saved.')
    
    parser.add_argument('--dataset_path', default = './datasets/mvtec_loco', type = str,
                        help = 'Dataset path.')
    
    parser.add_argument('--quantitative_folder', default='./results/loco/quantitatives_loco', type=str,
                        help='Path to the folder in which to save the quantitatives.')
    
    parser.add_argument('--qualitative_folder', default='./results/loco/qualitatives_loco', type=str,
                        help='Path to save qualitatives.')
    
    parser.add_argument('--produce_qualitatives', default=True, action='store_true',
                        help='Whether to produce qualitatives or not.')
    
    parser.add_argument('--class_name', default = "breakfast_box", type = str,
                        help = 'Category name.')
    
    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs to train.')
    
    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')

    parser.add_argument('--students_blocks', type=str, default='Full', choices=['Both_ViT', 'Both_LLM', 'Full'],
                        help='Inference scenario.')
    
    parser.add_argument('--label', default='qwen_assistant_generic_12_26_qwen_stud1', type=str,
                        help='Experiment label.')
    
    parser.add_argument('--vit_label', default='qwen_vit_9_15', type=str,
                        help='Label of experiment involving just ViT students.')
    
    parser.add_argument('--vit_layers', type=int, nargs=2, default=[9, 15],
                        help='The 2 ViT layers to be extracted')
    
    parser.add_argument('--llm_layers', type=int, nargs=2, default=[12, 26],
                        help='The 2 LLM layers to be extracted')

    args = parser.parse_args()
    infer(args)