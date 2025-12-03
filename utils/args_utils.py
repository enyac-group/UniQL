

def argparse_shared_options(parser):
    parser.add_argument(
        'model_repo', type=str,
        help='The model repository on huggingface hub, e.g., nvidia/Nemotron-H-8B-Base-8K'
    )
    parser.add_argument(
        '--pretrained_dir', type=str, default=None,
        help='The path to store the compressed and quantized model.'
        'Not storing if not provided. (default: None)'
    )
    parser.add_argument(
        '--quick_eval', action='store_true',
        help='Whether to do quick evaluation on the compressed model (default: False)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Whether to print the debug level information (default: False)'
    )
    return parser

def argparse_calibration_options(parser):
    ##### General Calibration Settings #####
    parser.add_argument(
        '--calib_data_repo', type=str, default="togethercomputer/RedPajama-Data-V2",
        help='The dataset repository for calibration data. (default: \"togethercomputer/RedPajama-Data-V2\")'
    )
    parser.add_argument(
        '--calib_data_num', type=int, default=512,
        help='Number of calibration data (default: 512). set -1 to use the whole dataset.'
    )
    parser.add_argument(
        '--calib_seqlen', type=int, default=1024,
        help='Maximum sequence length for calibration data (default: 1024)'
    )
    return parser

def argparse_quantize_options(parser):
    ##### Quantization Settings #####
    parser.add_argument(
        '--hadamard', action='store_true',
        help='Whether to use hadamard transform (default: False)'
    )
    parser.add_argument(
        '--gptq', action='store_true',
        help='Whether to apply GPTQ (default: False)'
    )
    parser.add_argument(
        '--w_bits', type=int, default=4,
        help='The weight quantization bits (default: 4)'
    )
    return parser

def argparse_compress_options(parser):
    ##### Low-rank Compression Settings #####
    parser.add_argument(
        '--layer_ratio_config', type=str, default=None,
        help='The json file to store/load the layer ratios for the compressing the model (default: None)'
    )
    parser.add_argument(
        '--uniform_ratio', type=float, default=1.0,
        help='The uniform ratio for the compressing the model (default: 1.0)'
    )
    return parser