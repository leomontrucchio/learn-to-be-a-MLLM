import argparse
import os
import torch
import torchvision
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
from utils_loco.config_loco import PROMPTS_CONFIG

from models.teacher import HDViTFeatureExtractor, HDLLMFeatureExtractor
from models.student import ResidualFeatureProjectionMLP, FeatureProjectionMLP, FeatureProjectionMLP_FullyTrainableSumHD

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- DeepSeek Resolution Helper ---
CANDIDATE_RESOLUTIONS = [
    [384, 384], [384, 768], [768, 384], [384, 1152], [1152, 384], [384, 1536], [1536, 384], [768, 768], [384, 1920],
    [1920,384], [384,2304], [2304,384], [768,1152], [1152,768], [384,2688], [2688,384], [384,3072], [3072,384],
    [768,1536], [1536,768], [384,3456], [3456,384], [1152, 1152]
  ]

def select_best_resolution(image_size, candidate_resolutions):
    original_width, original_height = image_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in candidate_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


# --- Run evaluation pipeline from mvtec ---
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
        "--num_parallel_workers", "10", 
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

    vit_label = args.vit_label
    llm_label = args.label

    # Dataloader.
    test_loader = get_hd_data_loader("test", args.class_name, args.dataset_path, batch_size=1)

    # Model Initialization
    students = {}
    teachers = {}
    check_dir = f'{args.checkpoint_folder}/{args.class_name}'

    print(f"Detecting dynamic grid size for {args.class_name}...")
    sample_batch = next(iter(test_loader))
    pil_imgs_sample, _ = sample_batch

    if args.students_blocks == 'Both_ViT':
        teachers['fe'] = HDViTFeatureExtractor(layers=[11, 15]).to(device, dtype=dtype).eval()
        with torch.no_grad():
            feat, _ = teachers['fe'](pil_imgs_sample)
        num_patches = feat.shape[1] * feat.shape[2]
        embed_dim = teachers['fe'].embed_dim
        
        students['backward_net'] = FeatureProjectionMLP_FullyTrainableSumHD(in_features=embed_dim, out_features=embed_dim, num_patches=num_patches, act_layer=torch.nn.GELU, reduction_factor=0.4).to(device, dtype=dtype)
        students['forward_net'] = FeatureProjectionMLP_FullyTrainableSumHD(in_features=embed_dim, out_features=embed_dim, num_patches=num_patches, act_layer=torch.nn.GELU, reduction_factor=0.4).to(device, dtype=dtype)

        students['forward_net'].load_state_dict(torch.load(os.path.join(check_dir, f'forward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'), map_location=device))
        students['backward_net'].load_state_dict(torch.load(os.path.join(check_dir, f'backward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'), map_location=device))

    elif args.students_blocks == 'Both_LLM':
        teachers['fe'] = HDLLMFeatureExtractor(conversation_template=PROMPTS_CONFIG[f'assistant_generic'], layers=[0, 1]).to(device).eval()
        with torch.no_grad():
            feat, _ = teachers['fe'](pil_imgs_sample)
        num_patches = feat.shape[1] * feat.shape[2]
        embed_dim = teachers['fe'].embed_dim

        students['backward_net'] = FeatureProjectionMLP_FullyTrainableSumHD(in_features=embed_dim, out_features=embed_dim, num_patches=num_patches, act_layer=torch.nn.SiLU, reduction_factor=0.4).to(device, dtype=dtype)
        students['forward_net'] = FeatureProjectionMLP_FullyTrainableSumHD(in_features=embed_dim, out_features=embed_dim, num_patches=num_patches, act_layer=torch.nn.SiLU, reduction_factor=0.4).to(device, dtype=dtype)

        students['forward_net'].load_state_dict(torch.load(os.path.join(check_dir, f'forward_net_{llm_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'), map_location=device))
        students['backward_net'].load_state_dict(torch.load(os.path.join(check_dir, f'backward_net_{llm_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'), map_location=device))

    elif args.students_blocks == 'Full':
        # Logica mista: ViT No PE, LLM Sum PE
        teachers['vit_fe'] = HDViTFeatureExtractor(layers=[11, 15]).to(device, dtype=dtype).eval()
        teachers['llm_fe'] = HDLLMFeatureExtractor(conversation_template=PROMPTS_CONFIG['assistant_generic'], layers=[0, 1]).to(device).eval()
        
        with torch.no_grad():
            feat_v, _ = teachers['vit_fe'](pil_imgs_sample)
            feat_l, _ = teachers['llm_fe'](pil_imgs_sample)
        
        num_patches_v = feat_v.shape[1] * feat_v.shape[2]
        num_patches_l = feat_l.shape[1] * feat_l.shape[2]

        students['vit_fw'] = FeatureProjectionMLP_FullyTrainableSumHD(in_features=teachers['vit_fe'].embed_dim, out_features=teachers['vit_fe'].embed_dim, num_patches=num_patches_v, act_layer=torch.nn.GELU, reduction_factor=0.4).to(device, dtype=dtype)
        students['vit_bw'] = FeatureProjectionMLP_FullyTrainableSumHD(in_features=teachers['vit_fe'].embed_dim, out_features=teachers['vit_fe'].embed_dim, num_patches=num_patches_v, act_layer=torch.nn.GELU, reduction_factor=0.4).to(device, dtype=dtype)
        
        students['llm_fw'] = FeatureProjectionMLP_FullyTrainableSumHD(in_features=teachers['llm_fe'].embed_dim, out_features=teachers['llm_fe'].embed_dim, num_patches=num_patches_l, act_layer=torch.nn.SiLU, reduction_factor=0.4).to(device, dtype=dtype)
        students['llm_bw'] = FeatureProjectionMLP_FullyTrainableSumHD(in_features=teachers['llm_fe'].embed_dim, out_features=teachers['llm_fe'].embed_dim, num_patches=num_patches_l, act_layer=torch.nn.SiLU, reduction_factor=0.4).to(device, dtype=dtype)

        vit_fw_path = os.path.join(check_dir, f'forward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        vit_bw_path = os.path.join(check_dir, f'backward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        llm_fw_path = os.path.join(check_dir, f'forward_net_{llm_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        llm_bw_path = os.path.join(check_dir, f'backward_net_{llm_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')

        students['vit_fw'].load_state_dict(torch.load(vit_fw_path, map_location=device, weights_only=False))
        students['vit_bw'].load_state_dict(torch.load(vit_bw_path, map_location=device, weights_only=False))
        students['llm_fw'].load_state_dict(torch.load(llm_fw_path, map_location=device, weights_only=False))
        students['llm_bw'].load_state_dict(torch.load(llm_bw_path, map_location=device, weights_only=False))

    else:
        raise ValueError("Invalid students_blocks option.")

    for model in students.values():
        model.eval()

    # Gaussian Blur Setup
    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device=device)/(w_l**2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device)/(w_u**2)

    base_output_dir = os.path.join(args.anomaly_maps_dir, args.class_name, 'test')

    # --- Inference Loop ---
    for batch_data in tqdm(test_loader, desc=f'Inferencing {args.class_name}'):
        
        # Unpack Data
        pil_imgs, img_path_list = batch_data
        img_path = img_path_list[0]
        
        # Get original dimensions
        orig_w, orig_h = pil_imgs[0].size
        
        # Re-calculate the DeepSeek Grid and Scaling
        # Target grid the processor chose
        best_w, best_h = select_best_resolution((orig_w, orig_h), CANDIDATE_RESOLUTIONS)
        
        # Calculate the actual dimensions the image was scaled to inside the tiles
        scale = min(best_w / orig_w, best_h / orig_h)
        resized_w = int(orig_w * scale)
        resized_h = int(orig_h * scale)

        with torch.no_grad():
            
            if args.students_blocks == 'Both_ViT' or args.students_blocks == 'Both_LLM':
                # Extraction
                f1, f2 = teachers['fe'](pil_imgs)
                B, H, W, C = f1.shape
                
                # Spatial reshape
                f1_input = f1.reshape(B, H*W, C)
                f2_input = f2.reshape(B, H*W, C)
                
                # Prediction
                p1 = students['backward_net'](f2_input).reshape(B, H, W, C)
                p2 = students['forward_net'](f1_input).reshape(B, H, W, C)
                
                # Anomaly maps
                m1 = 1 - torch.nn.functional.cosine_similarity(p1.float(), f1.float(), dim=-1)
                m2 = 1 - torch.nn.functional.cosine_similarity(p2.float(), f2.float(), dim=-1)
                combined_anomaly_map = (m1 * m2).unsqueeze(1)

            elif args.students_blocks == 'Full':
                v1, v2 = teachers['vit_fe'](pil_imgs)
                l1, l2 = teachers['llm_fe'](pil_imgs)
                
                # ViT
                Bv, Hv, Wv, Cv = v1.shape
                pv1 = students['vit_bw'](v2.reshape(Bv, Hv*Wv, Cv)).reshape(Bv, Hv, Wv, Cv)
                pv2 = students['vit_fw'](v1.reshape(Bv, Hv*Wv, Cv)).reshape(Bv, Hv, Wv, Cv)
                
                # LLM
                Bl, Hl, Wl, Cl = l1.shape
                pl1 = students['llm_bw'](l2.reshape(Bl, Hl*Wl, Cl)).reshape(Bl, Hl, Wl, Cl)
                pl2 = students['llm_fw'](l1.reshape(Bl, Hl*Wl, Cl)).reshape(Bl, Hl, Wl, Cl)

                # Map Calculation
                vit_fw_map = 1 - torch.nn.functional.cosine_similarity(pv2.float(), v2.float(), dim=-1)
                vit_bw_map = 1 - torch.nn.functional.cosine_similarity(pv1.float(), v1.float(), dim=-1)
                llm_fw_map = 1 - torch.nn.functional.cosine_similarity(pl2.float(), l2.float(), dim=-1)
                llm_bw_map = 1 - torch.nn.functional.cosine_similarity(pl1.float(), l1.float(), dim=-1)

                # Combine
                vit_comb = (vit_fw_map * vit_bw_map).unsqueeze(1)
                llm_comb = (llm_fw_map * llm_bw_map).unsqueeze(1)
                
                # Upsample LLM map to match ViT map size
                vit_h, vit_w = vit_comb.shape[-2:]
                llm_comb_upsampled = torch.nn.functional.interpolate(
                    llm_comb, size=(vit_h, vit_w), mode='bilinear', align_corners=False
                )
                
                combined_anomaly_map = vit_comb * llm_comb_upsampled

            # Upsample to Padded Image
            combined_anomaly_map = torch.nn.functional.interpolate(combined_anomaly_map, size=(best_h, best_w), mode='bilinear')

            # Gaussian Blur
            for _ in range(5):
                combined_anomaly_map = torch.nn.functional.conv2d(input=combined_anomaly_map, padding=pad_l, weight=weight_l)
            for _ in range(3):
                combined_anomaly_map = torch.nn.functional.conv2d(input=combined_anomaly_map, padding=pad_u, weight=weight_u)

            # Crop the Valid Scaled Region
            combined_anomaly_map = torchvision.transforms.functional.center_crop(combined_anomaly_map, [resized_h, resized_w])

            # Scale back to Original Resolution
            combined_anomaly_map = torch.nn.functional.interpolate(combined_anomaly_map, size=(orig_h, orig_w), mode='bilinear')

            # Save
            parent_dir = os.path.basename(os.path.dirname(img_path)) # ex. logical_anomalies
            file_name = os.path.basename(img_path) # ex. 000.png
            file_name_no_ext = os.path.splitext(file_name)[0] # ex. 000

            save_dir = os.path.join(base_output_dir, parent_dir)
            save_path = os.path.join(save_dir, f"{file_name_no_ext}.tiff")

            os.makedirs(save_dir, exist_ok=True)

            map_numpy = combined_anomaly_map.squeeze().cpu().detach().numpy()
            tifffile.imwrite(save_path, map_numpy)

            # Qualitative Plot
            if args.produce_qualitatives:
                save_qual_path = f'{args.qualitative_folder}/{args.students_blocks}_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs/{parent_dir}'
                os.makedirs(save_qual_path, exist_ok=True)

                # Ground Truth Loading
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

    parser.add_argument('--students_blocks', type=str, default='Both_LLM', choices=['Both_ViT', 'Both_LLM', 'Full'],
                        help='Inference scenario.')
    
    parser.add_argument('--label', default='hd_distilled_0.4', type=str,
                        help='Experiment label.')
    
    parser.add_argument('--vit_label', default='hd_res_1_5_1', type=str,
                        help='Label of experiment involving just ViT students.')

    args = parser.parse_args()
    infer(args)