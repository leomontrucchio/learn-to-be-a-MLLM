export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)


class_names=("breakfast_box" "juice_bottle" "pushpins" "screw_bag" "splicing_connectors")

# MODEL 1
for class_name in "${class_names[@]}"
    do
        python3 utils_loco/extract_qwen_features.py --class_name $class_name --students_blocks 'LLM' --layers 12 26
        python3 train_loco_qwen_pe.py --class_name $class_name --label 'qwen_llm_pe_fulltrainsum_12_26' --students_blocks 'Both_LLM' --layers 12 26
        rm -rf ./features/*
    done

for class_name in "${class_names[@]}"
    do
        python3 infer_loco_qwen_pe.py --class_name $class_name --label 'qwen_llm_pe_fulltrainsum_12_26' --students_blocks 'Both_LLM' --llm_layers 12 26
    done
python3 ./utils_loco/aggregate_metrics_loco.py --label 'qwen_llm_pe_fulltrainsum_12_26' --students_blocks 'Both_LLM'

# MODEL 2
for class_name in "${class_names[@]}"
    do
        python3 utils_loco/extract_qwen_features.py --class_name $class_name --students_blocks 'ViT' --layers 11 17
        python3 train_loco_qwen_pe.py --class_name $class_name --label 'qwen_vit_pe_fulltrainsum_11_17' --students_blocks 'Both_ViT' --layers 11 17
        rm -rf ./features/*
    done

for class_name in "${class_names[@]}"
    do
        python3 infer_loco_qwen_pe.py --class_name $class_name --label 'qwen_vit_pe_fulltrainsum_11_17' --vit_label 'qwen_vit_pe_fulltrainsum_11_17' --students_blocks 'Both_ViT' --vit_layers 11 17
    done
python3 ./utils_loco/aggregate_metrics_loco.py --label 'qwen_vit_pe_fulltrainsum_11_17' --students_blocks 'Both_ViT'


