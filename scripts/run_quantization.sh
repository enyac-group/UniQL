#!/bin/bash

# Check if model name argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name|ALL>"
    echo "Example models:"
    echo "  Original models examples:"
    echo "    Qwen/Qwen2.5-7B"
    echo "    meta-llama/Llama-3.1-8B"
    echo "    meta-llama/Llama-2-7b-hf"
    echo "    ibm-ai-platform/Bamba-9B-v2"
    echo "    nvidia/Nemotron-H-8B-Base-8K"
    echo "    ut-enyac/mamba2-8b-converted"
    echo "  UniQL models examples:"
    echo "    ut-enyac/Qwen2.5-7B-uniql-1.0"
    echo "    ut-enyac/Llama-3.1-8B-uniql-1.0"
    echo "    ut-enyac/Llama-2-7b-hf-uniql-1.0"
    echo "    ut-enyac/Bamba-9B-v2-uniql-1.0"
    echo "    ut-enyac/Nemotron-H-8B-Base-8K-uniql-1.0"
    echo "    ut-enyac/mamba2-8b-converted-uniql-1.0"
    exit 1
fi

MODEL_ARG=$1

# Function to run quantization for a single model
run_model() {
    local model=$1
    echo "Running quantization for: $model"
    # Extract the part after "/" and convert to lowercase
    model_compress_dir=$(echo "$model" | cut -d'/' -f2 | tr '[:upper:]' '[:lower:]')
    python -m quantize.quantize_model $model --calib_data_repo wikitext2 --calib_data_num 256 --calib_seqlen 2048 --pretrained_dir ./pretrained_models/ --gptq --w_bits 4 --hadamard
}

run_model $MODEL_ARG
echo "Completed quantization for: $MODEL_ARG"
