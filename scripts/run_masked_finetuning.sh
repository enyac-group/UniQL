#!/bin/bash
set -euo pipefail

# ---------------------------
# Model list
# ---------------------------
models=(
    "ut-enyac/Qwen2.5-7B-uniql-1.0"
    "ut-enyac/Llama-3.1-8B-uniql-1.0"
    "ut-enyac/Llama-2-7b-hf-uniql-1.0"
    "ut-enyac/Bamba-9B-v2-uniql-1.0"
    "ut-enyac/Nemotron-H-8B-Base-8K-uniql-1.0"
    "ut-enyac/mamba2-8b-converted-uniql-1.0"
)

# Default ratio list
ratios=("0.6" "0.65" "0.7" "0.75" "0.8" "0.85" "0.9" "0.95")

# ---------------------------
# Functions
# ---------------------------

run_get_layer_ratio() {
    local model=$1
    echo ">>> Generating layer-wise compression configs for: $model"
    
    python -m compress.get_layer_ratio "$model" \
        --ratio_list ${ratios[@]} \
        --calib_data_repo wikitext2 \
        --calib_data_num 128 \
        --calib_seqlen 2048 \
        --pretrained_dir pretrained_models/
}

run_recovery_finetune() {
    local model=$1

    echo ">>> Running recovery fine-tuning for: $model"
    
    python -m compress.recovery_finetune "$model" \
        --epochs 5 \
        --calib_data_repo yahma/alpaca-cleaned \
        --calib_data_num -1 \
        --calib_seqlen 256 \
        --pretrained_dir pretrained_models/ \
        --lora \
        --masked_rft
}

run_pipeline() {
    local model=$1
    echo "=============================================="
    echo ">>> Starting pipeline for $model"
    echo "=============================================="

    # Convert model name to lowercase
    local model_lc=$(basename "$model" | tr '[:upper:]' '[:lower:]')

    # Skip run_get_layer_ratio if the folder exists
    if [ ! -d "compress/outputs/$model_lc" ]; then
        echo ">>> folder compress/outputs/$model_lc does not exist"
        run_get_layer_ratio "$model"
    else
        echo ">>> Folder compress/outputs/$model_lc found, skipping get_layer_ratio."
    fi

    run_recovery_finetune "$model"

    echo ">>> Completed pipeline for: $model"
    echo "----------------------------------------------"
}

# ---------------------------
# Main logic
# ---------------------------

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name|ALL>"
    echo "Please run strcutural sorting: \`bash scripts/run_structured_sorting.sh <model_name>\` first"
    echo "Available models:"
    for model in "${models[@]}"; do
        echo "  $model"
    done
    echo "  ALL (runs all models)"
    exit 1
fi

MODEL_ARG=$1

if [[ "${MODEL_ARG^^}" = "ALL" ]]; then
    echo ">>> Running pipeline for ALL models..."
    for model in "${models[@]}"; do
        run_pipeline "$model"
    done
    echo ">>> All models completed!"
else
    if [[ " ${models[*]} " == *" $MODEL_ARG "* ]]; then
        run_pipeline "$MODEL_ARG"
    else
        echo "Error: Model '$MODEL_ARG' not found."
        echo "Available models:"
        for model in "${models[@]}"; do
            echo "  $model"
        done
        exit 1
    fi
fi