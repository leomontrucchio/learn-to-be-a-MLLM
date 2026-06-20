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
from sklearn.metrics import roc_auc_score
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

from utils.data_loader import get_data_loader
from utils.general_utils import set_seeds, extract_metrics_loco, siglip_denormalize, SinusoidalPositionalEmbedding2D
from utils.prompts import get_prompt
from utils.metrics_utils import calculate_au_pro

from models.teacher import ViTFeatureExtractor, LLMFeatureExtractor, QwenViTFeatureExtractor, QwenLLMFeatureExtractor
from models.student import ResidualFeatureProjectionMLP, FeatureProjectionMLP, FeatureProjectionBottleneckMLP

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_evaluation_pipeline_loco(args):
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
    
    glob_auc, glob_30, glob_10, glob_05, glob_01 = extract_metrics_loco('global', metrics)
    struct_auc, struct_30, struct_10, struct_05, struct_01 = extract_metrics_loco('structural_anomalies', metrics)
    logic_auc, logic_30, logic_10, logic_05, logic_01 = extract_metrics_loco('logical_anomalies', metrics)

    print(f"\nResults for {args.class_name}:")
    header = f"{'Type':<12} | {'I-AUROC':<8} | {'sPRO@30%':<8} | {'sPRO@10%':<8} | {'sPRO@5%':<8} | {'sPRO@1%':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(f"{'Global':<12} | {glob_auc:.3f}    | {glob_30:.3f}    | {glob_10:.3f}    | {glob_05:.3f}    | {glob_01:.3f}")
    print(f"{'Structural':<12} | {struct_auc:.3f}    | {struct_30:.3f}    | {struct_10:.3f}    | {struct_05:.3f}    | {struct_01:.3f}")
    print(f"{'Logical':<12} | {logic_auc:.3f}    | {logic_30:.3f}    | {logic_10:.3f}    | {logic_05:.3f}    | {logic_01:.3f}")
    print("-" * len(header))

    result_file_name = f'{quantitative_dir}/{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.md'
    
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
        'global_i_auroc': [glob_auc], 'global_spro_30': [glob_30], 'global_spro_10': [glob_10], 'global_spro_05': [glob_05], 'global_spro_01': [glob_01],
        'struct_i_auroc': [struct_auc], 'struct_spro_30': [struct_30], 'struct_spro_10': [struct_10], 'struct_spro_05': [struct_05], 'struct_spro_01': [struct_01],
        'logic_i_auroc': [logic_auc], 'logic_spro_30': [logic_30], 'logic_spro_10': [logic_10], 'logic_spro_05': [logic_05], 'logic_spro_01': [logic_01],
    }
    
    pd.DataFrame(results).to_csv(result_file_name.replace('md', 'csv'), index=False, sep=',')

def infer(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    vit_label = getattr(args, 'vit_label', args.label)
    llm_label = args.label
    pe_alpha = args.pe_alpha
    pe_dim = args.pe_dim

    test_loader, max_hw = get_data_loader(
        args.dataset_name, "test", class_name=args.class_name,
        img_size=args.img_size, dataset_path=args.dataset_path
    )

    prompt_template = None
    if args.students_blocks in ['Both_LLM', 'Full']:
        prompt_template = get_prompt(args.dataset_name, args.mllm, 'generic')
        if prompt_template is None and args.dataset_name == 'visa':
             prompt_template = get_prompt(args.dataset_name, args.mllm, 'grounding') # fallback

    students = {}
    teachers = {}
    pe_generators = {}
    check_dir = f'{args.checkpoint_folder}/{args.class_name}'

    use_pe = (args.mllm == 'deepseek' and args.dataset_name == 'loco')
    actual_pe_dim = pe_dim if use_pe else 0

    if args.mllm == 'qwen':
        qwen_model_name = "Qwen/Qwen2-VL-7B-Instruct"
        shared_model = Qwen2VLForConditionalGeneration.from_pretrained(qwen_model_name, dtype=torch.bfloat16, device_map="auto").eval()
        shared_processor = Qwen2VLProcessor.from_pretrained(qwen_model_name)
        act_layer = torch.nn.SiLU
    else:
        act_layer = torch.nn.SiLU if args.students_blocks in ['Both_LLM', 'Full'] else torch.nn.GELU

    if args.students_blocks == 'Both_ViT':
        if args.mllm == 'deepseek':
            teachers['fe'] = ViTFeatureExtractor(layers=args.vit_layers).to(device, dtype=dtype).eval()
            grid_size = args.img_size // teachers['fe'].patch_size
            embed_dim = teachers['fe'].embed_dim
        else:
            teachers['fe'] = QwenViTFeatureExtractor(model=shared_model, processor=shared_processor, layers=args.vit_layers).to(device, dtype=dtype).eval()
            FACTOR = 28
            new_h, new_w = round(args.img_size / FACTOR) * FACTOR, round(args.img_size / FACTOR) * FACTOR
            grid_size = new_h // 14
            embed_dim = shared_model.visual.embed_dim

        if use_pe:
            pe_generators['vit'] = SinusoidalPositionalEmbedding2D(pe_dim=pe_dim, size_h=grid_size).to(device, dtype=dtype)
        
        students['backward_net'] = FeatureProjectionMLP(in_features=embed_dim, out_features=embed_dim, pe_dim=actual_pe_dim, act_layer=act_layer, reduction_factor=0.4).to(device, dtype=dtype)
        students['forward_net'] = FeatureProjectionMLP(in_features=embed_dim, out_features=embed_dim, pe_dim=actual_pe_dim, act_layer=act_layer, reduction_factor=0.4).to(device, dtype=dtype)

        fwd_path = os.path.join(check_dir, f'forward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        bwd_path = os.path.join(check_dir, f'backward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        students['forward_net'].load_state_dict(torch.load(fwd_path, map_location=device, weights_only=False))
        students['backward_net'].load_state_dict(torch.load(bwd_path, map_location=device, weights_only=False))

    elif args.students_blocks == 'Both_LLM':
        if args.mllm == 'deepseek':
            teachers['fe'] = LLMFeatureExtractor(conversation_template=prompt_template, layer1_idx=args.llm_layers[0], layer2_idx=args.llm_layers[1]).to(device).eval()
            grid_size = 14
            embed_dim = teachers['fe'].embed_dim
        else:
            teachers['fe'] = QwenLLMFeatureExtractor(model=shared_model, processor=shared_processor, layers=args.llm_layers).to(device).eval()
            FACTOR = 28
            new_h, new_w = round(args.img_size / FACTOR) * FACTOR, round(args.img_size / FACTOR) * FACTOR
            grid_size = new_h // 28
            embed_dim = shared_model.config.hidden_size

        if use_pe:
            pe_generators['llm'] = SinusoidalPositionalEmbedding2D(pe_dim=pe_dim, size_h=grid_size).to(device, dtype=dtype)
        
        students['backward_net'] = FeatureProjectionMLP(in_features=embed_dim, out_features=embed_dim, pe_dim=actual_pe_dim, act_layer=torch.nn.SiLU, reduction_factor=0.4).to(device=device, dtype=dtype)
        students['forward_net'] = FeatureProjectionMLP(in_features=embed_dim, out_features=embed_dim, pe_dim=actual_pe_dim, act_layer=torch.nn.SiLU, reduction_factor=0.4).to(device=device, dtype=dtype)

        fwd_path = os.path.join(check_dir, f'forward_net_{llm_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        bwd_path = os.path.join(check_dir, f'backward_net_{llm_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')

        students['forward_net'].load_state_dict(torch.load(fwd_path, map_location=device, weights_only=False))
        students['backward_net'].load_state_dict(torch.load(bwd_path, map_location=device, weights_only=False))

    elif args.students_blocks == 'Full':
        if args.mllm == 'deepseek':
            teachers['vit_fe'] = ViTFeatureExtractor(layers=args.vit_layers).to(device, dtype=dtype).eval()
            teachers['llm_fe'] = LLMFeatureExtractor(conversation_template=prompt_template, layer1_idx=args.llm_layers[0], layer2_idx=args.llm_layers[1]).to(device).eval()
            grid_size_vit = args.img_size // teachers['vit_fe'].patch_size
            grid_size_llm = 14
            vit_embed_dim = teachers['vit_fe'].embed_dim
            llm_embed_dim = teachers['llm_fe'].embed_dim
        else:
            teachers['vit_fe'] = QwenViTFeatureExtractor(model=shared_model, processor=shared_processor, layers=args.vit_layers).to(device, dtype=dtype).eval()
            teachers['llm_fe'] = QwenLLMFeatureExtractor(model=shared_model, processor=shared_processor, layers=args.llm_layers).to(device).eval()
            FACTOR = 28
            new_h, new_w = round(args.img_size / FACTOR) * FACTOR, round(args.img_size / FACTOR) * FACTOR
            grid_size_vit = new_h // 14
            grid_size_llm = new_h // 28
            vit_embed_dim = shared_model.visual.embed_dim
            llm_embed_dim = shared_model.config.hidden_size

        if use_pe:
            pe_generators['vit'] = SinusoidalPositionalEmbedding2D(pe_dim=pe_dim, size_h=grid_size_vit).to(device, dtype=dtype)
            pe_generators['llm'] = SinusoidalPositionalEmbedding2D(pe_dim=pe_dim, size_h=grid_size_llm).to(device, dtype=dtype)

        students['vit_fw'] = FeatureProjectionMLP(in_features=vit_embed_dim, out_features=vit_embed_dim, pe_dim=actual_pe_dim, reduction_factor=0.4).to(device=device, dtype=dtype)
        students['vit_bw'] = FeatureProjectionMLP(in_features=vit_embed_dim, out_features=vit_embed_dim, pe_dim=actual_pe_dim, reduction_factor=0.4).to(device=device, dtype=dtype)
        
        students['llm_fw'] = FeatureProjectionMLP(in_features=llm_embed_dim, out_features=llm_embed_dim, pe_dim=actual_pe_dim, act_layer=torch.nn.SiLU, reduction_factor=0.4).to(device=device, dtype=dtype)
        students['llm_bw'] = FeatureProjectionMLP(in_features=llm_embed_dim, out_features=llm_embed_dim, pe_dim=actual_pe_dim, act_layer=torch.nn.SiLU, reduction_factor=0.4).to(device=device, dtype=dtype)

        vit_fw_path = os.path.join(check_dir, f'forward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        vit_bw_path = os.path.join(check_dir, f'backward_net_{vit_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        llm_fw_path = os.path.join(check_dir, f'forward_net_{llm_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
        llm_bw_path = os.path.join(check_dir, f'backward_net_{llm_label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')

        students['vit_fw'].load_state_dict(torch.load(vit_fw_path, map_location=device, weights_only=False))
        students['vit_bw'].load_state_dict(torch.load(vit_bw_path, map_location=device, weights_only=False))
        students['llm_fw'].load_state_dict(torch.load(llm_fw_path, map_location=device, weights_only=False))
        students['llm_bw'].load_state_dict(torch.load(llm_bw_path, map_location=device, weights_only=False))

    for model in students.values():
        model.eval()

    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device=device)/(w_l**2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device)/(w_u**2)

    if args.dataset_name == 'loco':
        base_output_dir = os.path.join(args.anomaly_maps_dir, args.class_name, 'test')
    else:
        predictions, gts = [], []
        image_labels, pixel_labels = [], []
        image_preds, pixel_preds = [], []
        inference_time = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    for batch in tqdm(test_loader, desc=f'Inferencing {args.class_name}'):
        if args.dataset_name == 'loco':
            pil_img, tensor_img, img_path_list = batch
            img_path = img_path_list[0]
        else:
            pil_img, tensor_img, gt, label, img_path_list = batch
            img_path = img_path_list[0]
            defect_class_str = img_path.split('/')[-3]

        image_name_str = img_path.split('/')[-1]
        file_name_no_ext = os.path.splitext(image_name_str)[0]
        
        orig_w, orig_h = pil_img[0].size
        target_size = args.img_size
        scale = min(target_size / orig_w, target_size / orig_h)
        resized_w = int(orig_w * scale)
        resized_h = int(orig_h * scale)
        
        with torch.no_grad():
            tensor_img = tensor_img.to(device, dtype=dtype)
            
            if args.dataset_name == 'visa':
                start_event.record()

            combined_anomaly_map = None

            if args.students_blocks == 'Both_ViT':
                input_data = pil_img if args.mllm == 'qwen' else tensor_img
                first_patch, second_patch = teachers['fe'](input_data)
                
                if use_pe:
                    pe = pe_generators['vit'](first_patch)
                    first_enriched = torch.cat([first_patch, pe * pe_alpha], dim=-1)
                    second_enriched = torch.cat([second_patch, pe * pe_alpha], dim=-1)
                else:
                    first_enriched, second_enriched = first_patch, second_patch

                pred_1st_patch = students['backward_net'](second_enriched)
                pred_2nd_patch = students['forward_net'](first_enriched)
                
                first_map = 1 - torch.nn.functional.cosine_similarity(pred_1st_patch.float(), first_patch.float(), dim=-1)
                second_map = 1 - torch.nn.functional.cosine_similarity(pred_2nd_patch.float(), second_patch.float(), dim=-1)
                
                combined_anomaly_map = (first_map * second_map).reshape(1, 1, grid_size, grid_size)

            elif args.students_blocks == 'Both_LLM':
                input_data = pil_img
                if args.mllm == 'qwen':
                    first_patch, second_patch = teachers['fe'](input_data, prompt_template)
                else:
                    first_patch, second_patch = teachers['fe'](input_data)
                
                if use_pe:
                    pe = pe_generators['llm'](first_patch)
                    first_enriched = torch.cat([first_patch, pe * pe_alpha], dim=-1)
                    second_enriched = torch.cat([second_patch, pe * pe_alpha], dim=-1)
                else:
                    first_enriched, second_enriched = first_patch, second_patch

                pred_1st_patch = students['backward_net'](second_enriched)
                pred_2nd_patch = students['forward_net'](first_enriched)
                
                first_map = 1 - torch.nn.functional.cosine_similarity(pred_1st_patch.float(), first_patch.float(), dim=-1)
                second_map = 1 - torch.nn.functional.cosine_similarity(pred_2nd_patch.float(), second_patch.float(), dim=-1)
                
                combined_anomaly_map = (first_map * second_map).reshape(1, 1, grid_size, grid_size)

            elif args.students_blocks == 'Full':
                input_data_vit = pil_img if args.mllm == 'qwen' else tensor_img
                vit_1st, vit_2nd = teachers['vit_fe'](input_data_vit)
                
                if args.mllm == 'qwen':
                    llm_1st, llm_2nd = teachers['llm_fe'](pil_img, prompt_template)
                else:
                    llm_1st, llm_2nd = teachers['llm_fe'](pil_img)
                
                if use_pe:
                    pe_vit = pe_generators['vit'](vit_1st) * pe_alpha
                    vit_1st_enr = torch.cat([vit_1st, pe_vit], dim=-1)
                    vit_2nd_enr = torch.cat([vit_2nd, pe_vit], dim=-1)
                    
                    pe_llm = pe_generators['llm'](llm_1st) * pe_alpha
                    llm_1st_enr = torch.cat([llm_1st, pe_llm], dim=-1)
                    llm_2nd_enr = torch.cat([llm_2nd, pe_llm], dim=-1)
                else:
                    vit_1st_enr, vit_2nd_enr = vit_1st, vit_2nd
                    llm_1st_enr, llm_2nd_enr = llm_1st, llm_2nd

                pred_vit_2nd = students['vit_fw'](vit_1st_enr)
                pred_vit_1st = students['vit_bw'](vit_2nd_enr)
                pred_llm_2nd = students['llm_fw'](llm_1st_enr)
                pred_llm_1st = students['llm_bw'](llm_2nd_enr)
                
                vit_fw_map = 1 - torch.nn.functional.cosine_similarity(pred_vit_2nd.float(), vit_2nd.float(), dim=-1)
                vit_bw_map = 1 - torch.nn.functional.cosine_similarity(pred_vit_1st.float(), vit_1st.float(), dim=-1)
                llm_fw_map = 1 - torch.nn.functional.cosine_similarity(pred_llm_2nd.float(), llm_2nd.float(), dim=-1)
                llm_bw_map = 1 - torch.nn.functional.cosine_similarity(pred_llm_1st.float(), llm_1st.float(), dim=-1)

                vit_comb = (vit_fw_map.reshape(1, 1, grid_size_vit, grid_size_vit) * vit_bw_map.reshape(1, 1, grid_size_vit, grid_size_vit))
                
                llm_fw_res = torch.nn.functional.interpolate(llm_fw_map.reshape(1, 1, grid_size_llm, grid_size_llm), size=(grid_size_vit, grid_size_vit), mode='bilinear', align_corners=False)
                llm_bw_res = torch.nn.functional.interpolate(llm_bw_map.reshape(1, 1, grid_size_llm, grid_size_llm), size=(grid_size_vit, grid_size_vit), mode='bilinear', align_corners=False)
                llm_comb = llm_fw_res * llm_bw_res
                
                combined_anomaly_map = vit_comb + llm_comb

            combined_anomaly_map = torch.nn.functional.interpolate(combined_anomaly_map, size=(target_size, target_size), mode='bilinear')

            for _ in range(5):
                combined_anomaly_map = torch.nn.functional.conv2d(input=combined_anomaly_map, padding=pad_l, weight=weight_l)
            for _ in range(3):
                combined_anomaly_map = torch.nn.functional.conv2d(input=combined_anomaly_map, padding=pad_u, weight=weight_u)

            combined_anomaly_map = torchvision.transforms.functional.center_crop(combined_anomaly_map, [resized_h, resized_w])
            combined_anomaly_map = torch.nn.functional.interpolate(combined_anomaly_map, size=(orig_h, orig_w), mode='bilinear')
            
            if args.dataset_name == 'visa':
                end_event.record()
                torch.cuda.synchronize()
                inference_time.append(start_event.elapsed_time(end_event))

            map_numpy = combined_anomaly_map.to(torch.float32).cpu().detach().numpy().squeeze()

            if args.dataset_name == 'loco':
                parent_dir = os.path.basename(os.path.dirname(img_path))
                save_dir = os.path.join(base_output_dir, parent_dir)
                save_path = os.path.join(save_dir, f"{file_name_no_ext}.tiff")
                os.makedirs(save_dir, exist_ok=True)
                tifffile.imwrite(save_path, map_numpy)

                if args.produce_qualitatives:
                    save_qual_path = f'{args.qualitative_folder}/{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs/{parent_dir}'
                    os.makedirs(save_qual_path, exist_ok=True)
                    if parent_dir == 'good':
                        gt_np = np.zeros((orig_h, orig_w))
                    else:
                        gt_folder_path = os.path.join(args.dataset_path, args.class_name, 'ground_truth', parent_dir, file_name_no_ext)
                        gt_files = glob.glob(os.path.join(gt_folder_path, "*.png"))
                        gt_combined = None
                        for f in gt_files:
                            mask = np.array(Image.open(f).convert('L'))
                            mask = (mask > 0).astype(np.float32)
                            gt_combined = mask if gt_combined is None else np.maximum(gt_combined, mask)
                        gt_np = gt_combined if gt_combined is not None else np.zeros((orig_h, orig_w))

                    _, axs = plt.subplots(1, 3, figsize=(7, 3))
                    img_vis = siglip_denormalize(tensor_img)
                    img_vis = torch.nn.functional.interpolate(img_vis, size=[max_hw, max_hw], mode='bilinear')
                    img_vis = torchvision.transforms.functional.center_crop(img_vis, [orig_h, orig_w])
                    axs[0].imshow(img_vis.squeeze().permute(1, 2, 0).to(torch.float32).cpu().detach().numpy())
                    axs[0].set_title('Input Image')
                    axs[1].imshow(gt_np, cmap='gray')
                    axs[1].set_title('Ground-truth')
                    axs[2].imshow(map_numpy, cmap=plt.cm.jet)
                    axs[2].set_title('Anomaly Map')
                    for ax in axs.flat: ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_qual_path, file_name_no_ext), dpi=256)
                    plt.close()
            else:
                # VISA specific collection
                gt_np = gt.squeeze().cpu().detach().numpy()
                gts.append(gt_np)
                predictions.append(map_numpy)
                image_labels.append(label)
                pixel_labels.extend(gt.flatten().cpu().detach().numpy())

                K = int((map_numpy.shape[0] * map_numpy.shape[1]) * 0.001)
                global_score = torch.topk(combined_anomaly_map.flatten(), k=K)[0].mean().to(torch.float32).cpu().detach().numpy()
                
                image_preds.append(global_score)
                pixel_preds.extend(map_numpy.flatten())

                if args.produce_qualitatives:
                    save_path = f'{args.qualitative_folder}/{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs/{defect_class_str}'
                    os.makedirs(save_path, exist_ok=True)

                    _, axs = plt.subplots(1, 3, figsize=(7, 3))
                    img = siglip_denormalize(tensor_img)
                    img = torch.nn.functional.interpolate(img, size=[max_hw, max_hw], mode='bilinear')
                    img = torchvision.transforms.functional.center_crop(img, gt.shape[-2:])
                    axs[0].imshow(img.squeeze().permute(1, 2, 0).to(torch.float32).cpu().detach().numpy())
                    axs[0].set_title('Input Image')
                    axs[1].imshow(gt_np, cmap='gray')
                    axs[1].set_title('Ground-truth')
                    axs[2].imshow(map_numpy, cmap=plt.cm.jet)
                    axs[2].set_title('Anomaly Map')
                    for ax in axs.flat: ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, image_name_str), dpi=256)
                    plt.close()

    print("Inference complete.")
    if args.dataset_name == 'loco':
        run_evaluation_pipeline_loco(args)
    else:
        # VisA metric calculation
        au_pros_q4, _, weights = calculate_au_pro(gts, predictions, weighted=False)
        q1, q2, q3 = np.quantile(weights, 0.25), np.quantile(weights, 0.5), np.quantile(weights, 0.75)
        
        au_pros_q3, _, _ = calculate_au_pro(gts, predictions, weighted=False, size_thr=(0, q3))
        au_pros_q2, _, _ = calculate_au_pro(gts, predictions, weighted=False, size_thr=(0, q2))
        au_pros_q1, _, _ = calculate_au_pro(gts, predictions, weighted=False, size_thr=(0, q1))

        pixel_rocauc = roc_auc_score(np.array(pixel_labels), np.array(pixel_preds))
        image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

        result_file_name = f'{args.quantitative_folder}/{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.md'
        
        print(f'Metrics for class {args.class_name} with {args.epochs_no}ep_{args.batch_size}bs')
        print("P-AUROC  |  I-AUROC")
        print(f'  {pixel_rocauc:.3f}   |   {image_rocauc:.3f}\n')

        print(" QUARTILE | AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% |")
        print(f'    Q4    |   {au_pros_q4[0]:.3f}   |   {au_pros_q4[1]:.3f}   |   {au_pros_q4[2]:.3f}  |   {au_pros_q4[3]:.3f}  |')
        print(f'    Q3    |   {au_pros_q3[0]:.3f}   |   {au_pros_q3[1]:.3f}   |   {au_pros_q3[2]:.3f}  |   {au_pros_q3[3]:.3f}  |')
        print(f'    Q2    |   {au_pros_q2[0]:.3f}   |   {au_pros_q2[1]:.3f}   |   {au_pros_q2[2]:.3f}  |   {au_pros_q2[3]:.3f}  |')
        print(f'    Q1    |   {au_pros_q1[0]:.3f}   |   {au_pros_q1[1]:.3f}   |   {au_pros_q1[2]:.3f}  |   {au_pros_q1[3]:.3f}  |')

        os.makedirs(args.quantitative_folder, exist_ok=True)

        with open(result_file_name, "w") as f:
            f.write(f'Metrics for class {args.class_name} with {args.epochs_no}ep_{args.batch_size}bs\n')
            f.write('Cumulative\nAUPRO@30% & AUPRO@10% & AUPRO@5% & AUPRO@1% & P-AUROC & I-AUROC\n')
            f.write(f'{au_pros_q4[0]:.3f} & {au_pros_q4[1]:.3f} & {au_pros_q4[2]:.3f} & {au_pros_q4[3]:.3f} & {pixel_rocauc:.3f} & {image_rocauc:.3f}\n')
            f.write(f'{au_pros_q3[0]:.3f} & {au_pros_q3[1]:.3f} & {au_pros_q3[2]:.3f} & {au_pros_q3[3]:.3f}\n')
            f.write(f'{au_pros_q2[0]:.3f} & {au_pros_q2[1]:.3f} & {au_pros_q2[2]:.3f} & {au_pros_q2[3]:.3f}\n')
            f.write(f'{au_pros_q1[0]:.3f} & {au_pros_q1[1]:.3f} & {au_pros_q1[2]:.3f} & {au_pros_q1[3]:.3f}\n')

        results = {
            'class_name': [args.class_name], 'inference_time': [np.mean(inference_time)], 
            'i_auroc': [image_rocauc], 'p_auroc': [pixel_rocauc],
            'aupro_30_q4': [au_pros_q4[0]], 'aupro_10_q4': [au_pros_q4[1]], 'aupro_05_q4': [au_pros_q4[2]], 'aupro_01_q4': [au_pros_q4[3]],
            'aupro_30_q3': [au_pros_q3[0]], 'aupro_10_q3': [au_pros_q3[1]], 'aupro_05_q3': [au_pros_q3[2]], 'aupro_01_q3': [au_pros_q3[3]],
            'aupro_30_q2': [au_pros_q2[0]], 'aupro_10_q2': [au_pros_q2[1]], 'aupro_05_q2': [au_pros_q2[2]], 'aupro_01_q2': [au_pros_q2[3]],
            'aupro_30_q1': [au_pros_q1[0]], 'aupro_10_q1': [au_pros_q1[1]], 'aupro_05_q1': [au_pros_q1[2]], 'aupro_01_q1': [au_pros_q1[3]],
        }
        pd.DataFrame(results).to_csv(result_file_name.replace('md', 'csv'), index=False, sep='&')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Inference Framework')

    parser.add_argument('--dataset_name', default='loco', type=str, choices=['loco', 'visa'],
                        help='Dataset name.')
    parser.add_argument('--mllm', type=str, default='deepseek', choices=['deepseek', 'qwen'],
                        help='Which MLLM to use as teacher.')
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
    parser.add_argument('--img_size', default = 384, type = int,
                        help = 'Square image resolution.')
    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')
    parser.add_argument('--students_blocks', type=str, default='Both_LLM', choices=['Both_ViT', 'Both_LLM', 'Full'],
                        help='Inference scenario.')
    parser.add_argument('--label', default='ds_llm_pe010', type=str,
                        help='Experiment label.')
    parser.add_argument('--vit_label', default='ds_vit_pe010', type=str,
                        help='Label of experiment involving just ViT students.')
    parser.add_argument('--llm_layers', type=int, nargs=2, default=[0, 1],
                        help='2 layers to be extracted')
    parser.add_argument('--vit_layers', type=int, nargs=2, default=[11, 15],
                        help='2 layers to be extracted')
    parser.add_argument('--pe_alpha', type=float, default=0.1,
                        help='Scaling factor for Positional Encoding')
    parser.add_argument('--pe_dim', type=int, default=128,
                        help='Positional encoding dimension to be concatenated')

    args = parser.parse_args()
    infer(args)
