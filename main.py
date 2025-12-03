import os
import re
import json
import logging
import argparse
from datetime import datetime

import torch

from eval_utils import eval_func
from modeling.build_models import build_model_and_tokenizer

from utils.logger_utils import set_logger 
from utils.reproduce_utils import set_deterministic
from utils.model_utils import profile_size
from utils.model_utils import model_name_and_type
from utils.args_utils import argparse_shared_options
from utils.args_utils import argparse_compress_options

logger = logging.getLogger(os.path.basename(__file__))

def main(args):

    model_name, model_type = model_name_and_type(args.model_repo)

    # apply layer-wise compression config
    layer_ratio = None
    if args.layer_ratio_config is not None:
        logger.info(f"Loading layer-wise compression config {args.layer_ratio_config}...")
        with open(args.layer_ratio_config, "r") as f:
            layer_ratio = json.load(f)
        for i, cfg in enumerate(layer_ratio):
            logger.debug(f"Layer {i}: {cfg}")
        match = re.search(r'ratio-([0-9]+(?:\.[0-9]+)?)', args.layer_ratio_config)
        ratio = float(match.group(1))
        logger.info(f"Get average ratio: {ratio}")

    model, tokenizer = build_model_and_tokenizer(model_repo=args.model_repo, pretrained_dir=args.pretrained_dir,
                                                 layer_ratio_config=layer_ratio, dtype=torch.bfloat16)
    model.eval()

    # get model size
    profile_size(model, model_type, batch_size=args.batch_size, prompt_len=1024)
    
    logs = {}
    if args.eval:
        logger.info(f"Evaluating result using lm_eval, task(s): {args.task_list}")
        lm_eval_results = eval_func(
            model, tokenizer, 
            model_type=model_type,
            batch_size=args.batch_size,
            max_length=2048, # default
            task_list=args.task_list, 
            fewshot=args.fewshot,
            limit=100 if args.testing else None
        )
        logs['lm_eval'] = lm_eval_results['results']
    if not args.eval:
        logger.warning("No task to run with, try `--eval --fewshot n`?")

    if args.log_dir:
        #logs['result'] = results['results']
        logs['args'] = vars(args)
        os.makedirs(args.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")  # Generate timestamp
        log_paths = os.path.join(args.log_dir, f"{model_name}_fp16_{timestamp}.json")
        with open(log_paths, 'w') as fp:
            logger.info(f"Saving result to {log_paths}")
            json.dump(logs, fp, indent=4)

if __name__ == '__main__':    
    set_deterministic(1234)
    parser = argparse_shared_options(argparse.ArgumentParser())
    parser = argparse_compress_options(parser)
    ##### General Evaluation Settings #####
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--eval', action='store_true', default=False,
        help='Whether to evaluate the few-shot performance. Task(s) specified by `--tasks_list` and `--fewshot`'
    )
    parser.add_argument(
        '--fewshot', type=int, default=0,
        help='Number of shots for few-shot evaluation (0 for zero-shot)'
    )
    parser.add_argument(
        '--task_list', type=lambda s: [item for item in s.split(',')], default=["lambada_openai"],
        help='Task to be evaled, e.g., --task_list lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande,mmlu,nq_open,squadv2'
    )
    parser.add_argument(
        '--testing', action='store_true',
        help='testing with decreased sample count'
    )
    parser.add_argument(
        '--log_dir', type=str,
        help='path to the json log file storing the result of lm_evan and quantization settingarg'
    )
    args = parser.parse_args()
    if args.verbose:
        set_logger(logger, logging.DEBUG)
    else:
        set_logger(logger, logging.INFO)
    main(args)