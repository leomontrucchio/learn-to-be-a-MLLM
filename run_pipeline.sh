#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Example for LOCO with DeepSeek
echo "Running DeepSeek on LOCO dataset..."
class_names_loco=("breakfast_box" "juice_bottle" "pushpins" "screw_bag" "splicing_connectors")

for class_name in "${class_names_loco[@]}"
do
    echo "Training LOCO class (LLM students): $class_name"
    python3 train.py --dataset_name loco --mllm deepseek --class_name $class_name --students_blocks 'Both_LLM' --layers 0 1 --label 'ds_llm_pe010'
done

for class_name in "${class_names_loco[@]}"
do
    echo "Inferencing LOCO class (with LLM students): $class_name"
    python3 infer.py --dataset_name loco --mllm deepseek --class_name $class_name --students_blocks 'Both_LLM' --llm_layers 0 1 --label 'ds_llm_pe010'
done

# Example for VisA with Qwen
echo "Running Qwen on VisA dataset..."
class_names_visa=("candle" "capsules" "macaroni" "pcb1" "pcb2")

for class_name in "${class_names_visa[@]}"
do
    echo "Training VisA class (ViT students): $class_name"
    python3 train.py --dataset_name visa --mllm qwen --class_name $class_name --students_blocks 'Both_ViT' --layers 11 17 --label 'qwen_vit'
    echo "Training VisA class (LLM students): $class_name"
    python3 train.py --dataset_name visa --mllm qwen --class_name $class_name --students_blocks 'Both_LLM' --layers 12 26 --label 'qwen_llm'
done

for class_name in "${class_names_visa[@]}"
do
    echo "Inferencing VisA class: $class_name"
    python3 infer.py --dataset_name visa --mllm qwen --class_name $class_name --students_blocks 'Full' --llm_layers 12 26 --vit_layers 11 17 --vit_label 'qwen_vit' --label 'qwen_llm'
done
