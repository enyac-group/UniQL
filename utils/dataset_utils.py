import logging
from typing import Union
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets

"""
A dedicated helper to manage templates and prompt building.
"""

alpaca_template = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"    
}

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = ""):
        if template_name == 'alpaca':
            self.template = alpaca_template
        else:
            raise ValueError(f"Unknown template name: {template_name}")
        logging.debug(
            f"Using prompt template {template_name}: {self.template['description']}"
        )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        # logging.debug(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class ZeroPrompter(object):

    def __init__(self):
        logging.debug(
            f"Without using prompt template!"
        )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if instruction[-1] == '.':
            instruction = instruction[:-1] + ':'
        if instruction[-1] not in ['.', ':', '?', '!']:
            instruction = instruction + ':'
        instruction += ' '

        if input:
            if input[-1] not in ['.', ':', '?', '!']:
                input = input + '.'
            res = instruction + input
        else:
            res = instruction
        if label:
            res = f"{res} {label}"
        # logging.debug(res)
        return res

    def get_response(self, output: str) -> str:
        return output.strip()


def build_dataloader(dataset_repo, tokenizer, batch_size=8, num_sample=512, max_length=128,
                     columns=["input_ids", "attention_mask"], num_valid_split=0,
                     enable_instruct_prompting=True, instruct_output_length=-1, torch_dataloader=True):
    # Load dataset and tokenize
    if dataset_repo == "togethercomputer/RedPajama-Data-V2":
        dataset = load_dataset("togethercomputer/RedPajama-Data-V2", name="sample", split="train")  # redpajama only has training set

        def tokenize_fn(data, tokenizer):
            tokenized = tokenizer(data["raw_content"], truncation=True, padding="max_length", max_length=max_length)
            if "labels" in columns:
                tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
    
    elif dataset_repo == "monology/pile-uncopyrighted":
        dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train") # pile only has training set

        def tokenize_fn(example, tokenizer):
            tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)
            if "labels" in columns:
                tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
    
    elif dataset_repo == "yahma/alpaca-cleaned":
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        if instruct_output_length > 0:
            dataset = dataset.filter(lambda example: len(example["output"]) >= instruct_output_length)
        prompter = Prompter("alpaca") if enable_instruct_prompting else ZeroPrompter()

        def tokenize_fn(data, tokenizer):
            prompts = [
                prompter.generate_prompt(inst, inp, out)
                for inst, inp, out in zip(data["instruction"], data["input"], data["output"])
            ]
            tokenized = tokenizer(prompts, truncation=True, padding="max_length", max_length=max_length)
            if "labels" in columns:
                tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
    elif dataset_repo == "wikitext2":
        # "wikitext-2-v1" is preprocessed, punctuation-normalized and tokenized.
        # "wikitext-2-raw-v1" is non-preprocessed, raw text.
        # Failed to perform Cholesky decomposition, not sure why 
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        def tokenize_fn(example, tokenizer):
            tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)
            if "labels" in columns:
                tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
    else:
        raise ValueError(f"Unsupported dataset: {dataset_repo}")
    
    dataset.shuffle()
    tokenizer.pad_token = tokenizer.eos_token # we have to pad here because we put all the data into a single batch
    sampled_dataset = dataset
    if num_sample is not None and num_sample > 0:
        sampled_dataset = sampled_dataset.select(range(num_sample))  # limit for demo
    preprocess_fn = partial(tokenize_fn, tokenizer=tokenizer)
    sampled_dataset = sampled_dataset.map(preprocess_fn, batched=True)
    sampled_dataset.set_format(type="torch", columns=columns)

    if num_valid_split > 0:
        sampled_trainval = sampled_dataset.train_test_split(test_size=num_valid_split, shuffle=True)
        train_loader = sampled_trainval["train"]
        valid_loader = sampled_trainval["test"]
    else:
        train_loader = sampled_dataset
        valid_loader = None

    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True) if torch_dataloader else train_loader
    valid_loader = DataLoader(valid_loader, batch_size=batch_size, shuffle=False) if torch_dataloader else valid_loader
    return train_loader, valid_loader

def build_combined_dataloader(tokenizer, batch_size=8, num_sample=512, max_length=128):

    def tokenize_redpajama(example, tokenizer):
        return tokenizer(example["raw_content"], truncation=True, padding="max_length", max_length=max_length)
    redpajama = load_dataset("togethercomputer/RedPajama-Data-V2", name="sample", split="train").select(range(num_sample // 2))
    redpajama = redpajama.map(partial(tokenize_redpajama, tokenizer=tokenizer), batched=True)

    def tokenize_pile(example, tokenizer):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)
    pile = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train").select(range(num_sample // 2))
    pile = pile.remove_columns(["meta"])
    pile = pile.map(partial(tokenize_pile, tokenizer=tokenizer), batched=True)
    
    combined_dataset = concatenate_datasets([redpajama, pile])
    combined_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    # Create DataLoader
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 


def get_ptb_new(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name, tokenizer, nsamples=128, seed=0, seqlen=2048,
    model=''
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    elif 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model, tokenizer)
        return get_c4(nsamples, seed, seqlen, model, tokenizer)
    else:
        raise NotImplementedError(f"Dataset {name} not supported, only support [wikitext2, c4]")