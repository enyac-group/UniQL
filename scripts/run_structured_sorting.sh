#!/bin/bash

# Check if model name argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name|ALL>"
    echo "Available models:"
    echo "  Qwen/Qwen2.5-7B"
    echo "  meta-llama/Llama-3.1-8B"
    echo "  meta-llama/Llama-2-7b-hf"
    echo "  ibm-ai-platform/Bamba-9B-v2"
    echo "  nvidia/Nemotron-H-8B-Base-8K"
    echo "  ut-enyac/mamba2-8b-converted"
    echo "  ALL (runs all models)"
    exit 1
fi

MODEL_ARG=$1

# Define all available models
models=("Qwen/Qwen2.5-7B" "meta-llama/Llama-3.1-8B" "meta-llama/Llama-2-7b-hf" "ibm-ai-platform/Bamba-9B-v2" "nvidia/Nemotron-H-8B-Base-8K" "ut-enyac/mamba2-8b-converted")

# Function to run structured sorting for a single model
run_model() {
    local model=$1
    echo "Running structured sorting for: $model"
    # Extract the part after "/" and convert to lowercase
    model_compress_dir=$(echo "$model" | cut -d'/' -f2 | tr '[:upper:]' '[:lower:]')
    python -m compress.compress_model  $model --calib_data_repo yahma/alpaca-cleaned --calib_data_num 128 --calib_seqlen 2048 --pretrained_dir pretrained_models/
}

# Check if user wants to run all models
if [ "$MODEL_ARG" = "ALL" ]; then
    echo "Running structured sorting for ALL models..."
    for model in ${models[@]}; do
        run_model $model
        echo "Completed analysis for: $model"
        echo "----------------------------------------"
    done
    echo "All models completed!"
else
    # Check if the specified model exists in the list
    model_found=false
    for model in ${models[@]}; do
        if [ "$model" = "$MODEL_ARG" ]; then
            model_found=true
            break
        fi
    done
    
    if [ "$model_found" = true ]; then
        run_model $MODEL_ARG
        echo "Completed structured sorting for: $MODEL_ARG"
    else
        echo "Error: Model '$MODEL_ARG' not found in available models."
        echo "Available models:"
        for model in ${models[@]}; do
            echo "  $model"
        done
        echo "Use 'ALL' to run all models."
        exit 1
    fi
fi