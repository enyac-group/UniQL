
# UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs


[Hung-Yueh Chiang](https://hychiang.info/),
[Chi-Chih Chang](https://ccchang.info/),
[Yu-Chen Lu](https://www.linkedin.com/in/%E6%98%B1%E8%BE%B0-%E5%91%82-8314b5227/),
[Chien-Yu Lin](https://cylinbao.github.io/),
[Kai-Chiang Wu](https://people.cs.nycu.edu.tw/~kcw/),
[Mohamed S. Abdelfattah](https://www.mohsaied.com/),
[Diana Marculescu](https://users.ece.utexas.edu/~dianam/)

<!-- ![UniQL CI](https://github.com/enyac-group/UniQL/actions/workflows/uniql-ci.yml/badge.svg) -->
<!-- [![UniQL arXiv](https://img.shields.io/badge/UniQL-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2503.22879) -->
<!-- [![UniQL Page](https://img.shields.io/badge/UniQL-Website-orange)](https://hychiang.info/projects/uniql/) -->

<p align="center">
  <img src="misc/logo.png" alt="UniQL" width="200"/>
</p>


## Setup


### Hardware requirements
- NVIDIA GPU Ampere architecture or above

### Software requirements
- CUDA 12.6
- python 3.12
- CMAKE version 3.22.1 or above
- PyTorch 2.7.1
- Triton 3.3.1

### Clone UniQL
- To clone the repository with all submodules:
```bash
git clone --recurse-submodules git@github.com:enyac-group/UniQL.git
cd UniQL
# or
git clone git@github.com:enyac-group/UniQL.git
cd UniQL
git submodule update --init --recursive
```

- Run in docker (optional)
As our implementation includes customized CUDA kernels and depends on specific CUDA version, user may optionally run our code in docker. To build the docker image, run the following command:
```bash
cd docker
./build_docker.sh
```

After building the docker image, user can run the docker container with the following command:
```bash
./run.sh
```

- Create UniQL conda environment
```bash
cd UniQL
conda create -n UniQL python=3.12
conda activate UniQL
pip install -r requirements.txt
```

### Build and install 3rd-party libraries

- Install `lm-evaluation-harness`:
```bash
pip install 3rdparty/lm-evaluation-harness
```

- Install `fast-hadamard-transform`:
```bash
# set force build to include 12N, 40N from the newer commit
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE
pip install 3rdparty/fast-hadamard-transform  --no-build-isolation
```

- Install `peft` from our commit:
```bash
# we fix peft for mamba blocks
pip install 3rdparty/peft  --no-build-isolation
```

- (Optional) Build `causal_conv1d` from source (or you can install from the prebuilt wheels):
```bash
# build from the local clone
pip install 3rdparty/causal-conv1d --no-build-isolation
```

- (Optional) Build `mamba_ssm` from source (or you can install from the prebuilt wheels):
```bash
# build from the local clone
export MAMBA_FORCE_BUILD=TRUE
pip install 3rdparty/mamba --no-build-isolation
```

- (Optional) Build `flash_attn` from source (or you can install from the prebuilt wheels):
```bash
# build from the local clone
cd 3rdparty/flash-attention
MAX_JOBS=16 python setup.py install
```

### Build and install UniQL
```
pip install -e .  --no-build-isolation
```


## Supported models

### Transformers
 - [x] Qwen/Qwen2.5-7B
 - [x] Qwen/Qwen2.5-7B-Instruct
 - [x] meta-llama/Llama-3.1-8B
 - [x] meta-llama/Llama-3.1-8B-Instruct
 - [x] meta-llama/Llama-2-7b-hf

### Mamba-Transformer hybrids
 - [x] ibm-ai-platform/Bamba-9B-v2
 - [x] nvidia/Nemotron-H-8B-Base-8K

### Mamba
 - [x] mamba2-8b (convert from nvidia/mamba2-8b-3t-4k)


## Structured weight-sorting
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_structured_sorting.sh meta-llama/Llama-3.1-8B
```
For this example, the sorted model will be stored at this path `pretrained_models/ut-enyac/Llama-3.1-8B-uniql-1.0`. `1.0` here means we don't prune the weights and only sort the weights in the model. Please see `compress/compress_models.py` for more details.

## Recovery Masked LoRA fine-tuning
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_masked_finetuning.sh ut-enyac/Llama-3.1-8B-uniql-1.0
```
If the pruning ratios are not found under `compress/output/{model name}`, the script will run `compress/get_layer_ratios.py` first. For example, the layerwise pruning ratios will be stored at `compress/outputs/llama-3.1-8b-uniql-1.0/`.

The fine-tuned model will be stored at this path `pretrained_models/ut-enyac/Llama-3.1-8B-uniql-1.0-masked-lora-rft`. 


## Quantize UniQL models
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_quantization.sh ut-enyac/Llama-3.1-8B-uniql-1.0-masked-lora-rft
```
For this example, the low-rank model will be stored at this path `pretrained_models/ut-enyac/Llama-3.1-8B-uniql-1.0-masked-lora-rft-w4a16`


## Evaluation

### Zero-shot evaluation
```bash
# standard zero-shot
CUDA_VISIBLE_DEVICES=0 python main.py ut-enyac/Llama-3.1-8B-uniql-1.0-masked-lora-rft-w4a16 --batch_size 16 --eval --fewshot 0 --task_list hellaswag,arc_easy,arc_challenge,piqa,winogrande --pretrained_dir ./pretrained_models/  --layer_ratio_config ./compress/outputs/llama-3.1-8b-uniql-1.0/layerwise_eps-0.1_ratio-0.85.json
```

### Five-shot evaluation
```bash
# we use batch size 8 to save memory
CUDA_VISIBLE_DEVICES=0 python main.py ut-enyac/Llama-3.1-8B-uniql-1.0-masked-lora-rft-w4a16 --batch_size 8 --eval --fewshot 5 --task_list mmlu --pretrained_dir ./pretrained_models/ --layer_ratio_config ./compress/outputs/llama-3.1-8b-uniql-1.0/layerwise_eps-0.1_ratio-0.85.json
```

## Profiling

- To profile model size, use `--size`:
```bash
python profile_model.py ut-enyac/Llama-3.1-8B-uniql-1.0-masked-lora-rft-w4a16 --size --batch_size 1 --pretrained_dir pretrained_models/
```

- To profile prefilling latency (i.e., time-to-first-token), use `--ttft`:
```bash
# ttft does not support --cache_graph
python profile_model.py ut-enyac/Llama-3.1-8B-uniql-1.0-masked-lora-rft-w4a16 --ttft --batch_size 1 --prompt_len 1024 --pretrained_dir pretrained_models/
```

- To profile generation latency (i.e., decoding phase, time-per-output-token), use `--tpot`:
```bash
python profile_model.py ut-enyac/Llama-3.1-8B-uniql-1.0-masked-lora-rft-w4a16 --tpot --batch_size 1 --cache_graph --prompt_len 1024 --pretrained_dir pretrained_models/
```

- To profile end-to-end latency (i.e., prefilling + decoding time, time-to-last-token), use `--ttlt`:
```bash
# profiling ttlt might take more time, so we report the average latency of 20 profilings
python profile_model.py ut-enyac/Llama-3.1-8B-uniql-1.0-masked-lora-rft-w4a16 --ttlt --batch_size 1 --cache_graph --prompt_len 1024 --gen_len 1024 --repeats 20 --pretrained_dir pretrained_models/
```

Please use `--help` to see more profiling configurations.

## Mamba2-8B

### Convert Nvidia Mamba2-8B to HuggingFace

Download the checkpoint using `huggingface-cli`
```bash
huggingface-cli download nvidia/mamba2-8b-3t-4k --local-dir ./pretrained_models/mamba2-8b-3t-4k
```
After downloading, you will have the directory `./pretrained_models/mamba2-8b-3t-4k` having a structure like this
```bash
├── latest_checkpointed_iteration.txt
├── mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model (This is tokenizer)
├── README.md
└── release
    └── mp_rank_00
        └── model_optim_rng.pt (This is weights)
```
+ Run the conversion scripts to get the model directory
```bash
python convert_mamba2_8b_to_hf.py \
./pretrained_models/mamba2-8b-3t-4k/release/mp_rank_00/model_optim_rng.pt \
./pretrained_models/mamba2-8b-3t-4k/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
--model_save_path ./pretrained_models/ut-enyac/mamba2-8b-converted
```


<!-- ## Citation
```
@inproceedings{chiang2025uniql,
  title = {UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs},
  author = {Chiang, Hung-Yueh and Chang, Chi-Chih and Lu, Yu-Chen and Lin, Chien-Yu and Wu, Kai-Chiang and Abdelfattah, Mohamed S. and Marculescu, Diana},
  booktitle = {Forty-Second International Conference on Machine Learning (ICML)},
  year = {2025}
}
```` -->
