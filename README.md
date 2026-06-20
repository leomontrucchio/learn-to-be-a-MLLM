# Learning to be a Multimodal LLM: Investigating Visual-Textual Inconsistency for Industrial Anomaly Detection

This repository contains the codebase developed for my Master's Thesis.

The framework leverages the powerful visual and linguistic understanding of state-of-the-art MLLMs as "Teachers" to perform complex anomaly detection.

## Key Features

*   **Unified Architecture**: A clean interface for both training (`train.py`) and inference (`infer.py`) across different datasets.
*   **Multi-Model Support**: Native integration with both **DeepSeek-VL2** and **Qwen2-VL** architectures.
*   **Complex Anomaly Detection**: Effectively handles logical and structural anomalies on the **MVTec LOCO AD** dataset, as well as standard structural defects on the **VisA** dataset.
*   **Adaptive Spatial Integration**: Dynamically handles Positional Encoding (PE). It enables PE for DeepSeek-VL2 to ground spatial anomalies on LOCO, while automatically disabling it for Qwen2-VL to avoid conflicts with its native M-ROPE 2D spatial embeddings.

## Project Structure

*   `train.py`: Unified script for training the Student MLP networks on anomaly-free images.
*   `infer.py`: Unified script for inference, anomaly map generation, and performance evaluation.
*   `models/`: Contains the architecture wrappers for the Teacher MLLMs and the Student MLPs.
*   `utils/`: Core utilities including data loaders, prompt definitions, metric calculations (AUROC, AUPRO, sPRO), and general helpers.
*   `run_pipeline.sh`: A ready-to-use shell script containing practical examples for running the pipeline on both datasets.
*   `mvtec_loco_ad_evaluation/`: The official metric library used for LOCO AD evaluation.

## Installation & Setup

1.  **Environment**: Ensure you have Python 3.9+ and install the required dependencies (PyTorch, Transformers, WandB, etc.).
    ```bash
    pip install torch torchvision transformers wandb pandas numpy scipy
    ```
2.  **MLLM Dependencies**: If using DeepSeek-VL2 or Qwen2-VL, ensure you install their specific library dependencies as detailed in their respective official documentation (`deepseek-vl2` and `Qwen2-VL`).
3.  **Datasets**: Download the **MVTec LOCO AD** and **VisA** datasets and place them in the `./datasets` directory (e.g., `./datasets/mvtec_loco` and `./datasets/visa`).

## Usage

### 1. Training (`train.py`)
The training script distills knowledge from the Teacher blocks into the Student networks using only anomaly-free images.

**Key Arguments:**
*   `--dataset_name`: `loco` or `visa`
*   `--mllm`: `deepseek` or `qwen`
*   `--students_blocks`: Defines which blocks to train (`Both_ViT` for Vision blocks, `Both_LLM` for Language blocks).
*   `--layers`: The layer indices to extract features from (e.g., `0 1` or `11 15`).
*   `--label`: Identifier used for saving checkpoints.

### 2. Inference (`infer.py`)
The inference script evaluates the trained students and computes metrics. It can also combine both Vision and Language student predictions using the `Full` mode.

**Key Arguments:**
*   `--students_blocks`: Can be `Both_ViT`, `Both_LLM`, or `Full` (combines both blocks for a comprehensive anomaly map).
*   `--llm_layers` / `--vit_layers`: The layer indices used during the training phase.
*   `--label` / `--vit_label`: The corresponding labels used to load the saved checkpoints.

### Examples

A comprehensive example of how to execute the framework is provided in `run_pipeline.sh`.

**Example 1: DeepSeek-VL2 on LOCO (LLM Students)**
```bash
# Train the LLM Students
python3 train.py --dataset_name loco --mllm deepseek --class_name breakfast_box --students_blocks 'Both_LLM' --layers 0 1 --label 'ds_llm_pe010'

# Evaluate the LLM Students
python3 infer.py --dataset_name loco --mllm deepseek --class_name breakfast_box --students_blocks 'Both_LLM' --llm_layers 0 1 --label 'ds_llm_pe010'
```

**Example 2: Qwen2-VL on VisA (Full Architecture)**
```bash
# 1. Train the ViT Students
python3 train.py --dataset_name visa --mllm qwen --class_name candle --students_blocks 'Both_ViT' --layers 11 17 --label 'qwen_vit'

# 2. Train the LLM Students separately
python3 train.py --dataset_name visa --mllm qwen --class_name candle --students_blocks 'Both_LLM' --layers 12 26 --label 'qwen_llm'

# 3. Evaluate using the 'Full' configuration combining both ViT and LLM checkpoints
python3 infer.py --dataset_name visa --mllm qwen --class_name candle --students_blocks 'Full' --llm_layers 12 26 --vit_layers 11 17 --vit_label 'qwen_vit' --label 'qwen_llm'
```

## Metrics & Evaluation
The framework handles dataset-specific metrics automatically and saves CSV reports in the `results/` folder:
*   **MVTec LOCO AD**: Outputs standard Global, Structural, and Logical metrics including **I-AUROC** and **sPRO**.
*   **VisA**: Computes standard defect detection spatial metrics including **P-AUROC**, **I-AUROC**, and **AUPRO** at multiple integration limits and quantiles of image dimension.