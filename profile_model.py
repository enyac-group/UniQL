import os
import re
import math
import time
import gzip
import json
import socket
import logging
import argparse
from functools import partial
from typing import Optional
from multiprocessing import Process, Event, Manager
from datetime import datetime

# to profile gpu power
try:
    from pynvml import *
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    nvmlDeviceGetPowerUsage(handle)
    _NVML_AVAILABLE = True
except Exception as e:
    _NVML_AVAILABLE = False

try:
    from jtop import jtop
    _JTOP_AVAILABLE = True
except Exception:
    _JTOP_AVAILABLE = False

import torch
import torch.nn as nn
from torch.autograd.profiler import record_function
from hta.trace_analysis import TraceAnalysis

from utils.logger_utils import set_logger
from utils.reproduce_utils import set_deterministic
from utils.model_utils import contains_wxa16
from utils.model_utils import get_wxa16_number
from utils.model_utils import model_name_and_type
from utils.args_utils import argparse_shared_options
from utils.args_utils import argparse_compress_options

from modeling.build_models import build_model_and_tokenizer
from modeling.model_helper_registry import ModelHelperRegistry

logger = logging.getLogger(os.path.basename(__file__))


def get_real_gpu_index():
    """
    Returns:
        logical_id: index used by torch.cuda (0..n-1 under CUDA_VISIBLE_DEVICES)
        physical_id: global index as seen by nvidia-smi
        pci_bus_id: unique PCI identifier (e.g. '00000000:65:00.0')
    """
    logical_id = torch.cuda.current_device()
    if _NVML_AVAILABLE:
        nvmlInit()
        try:
            handle = nvmlDeviceGetHandleByIndex(logical_id)
            pci_info = nvmlDeviceGetPciInfo(handle)
            # Handle both bytes and str
            pci_bus_id = pci_info.busId.decode("utf-8") if isinstance(pci_info.busId, bytes) else pci_info.busId
        finally:
            nvmlShutdown()

        # Map logical → physical using CUDA_VISIBLE_DEVICES
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible is not None:
            visible_list = [v.strip() for v in visible.split(",") if v.strip()]
            physical_id = int(visible_list[logical_id])
        else:
            physical_id = logical_id
    elif _JTOP_AVAILABLE:
        pci_bus_id = "unknown"
        physical_id = logical_id
    else:
        raise RuntimeError("No GPU available")

    return logical_id, physical_id, pci_bus_id

"""
{
  "rail": { ... },   # per-rail sensor readings from the INA3221 chips, lists individual voltage rails that power different functional blocks of the SoC.
  "tot": { ... }     # total board (VDD_IN) power. Represents the total input power drawn by the Jetson module
}
"""
# Rails that often carry GPU power on Jetson
GPU_RAIL_CANDIDATES = [
    "VDD_GPU_SOC",
    "VDD_SYS_GPU",
    "VDD_GPU",
    "GPU",
]

def find_nano_gpu_rail_name(rail_dict):
    """
    Given power_data['rail'] (a dict of {rail_name: {power, voltage, current}}),
    return the best-matching rail name for GPU power.
    On the Jetson Orin Nano:
        •	You cannot directly separate GPU-only power, because it shares the regulator with CPU and CV blocks (VDD_CPU_GPU_CV).
        •	The closest approximation to “GPU power” is VDD_CPU_GPU_CV, but it also rises when CPU workloads are active.
    """
    # 1) Exact/known names first
    for name in GPU_RAIL_CANDIDATES:
        if name in rail_dict:
            return name
    # 2) Fuzzy match: any rail name that contains 'GPU'
    for name in rail_dict: # we get "VDD_CPU_GPU_CV" here
        if "GPU" in name.upper():
            return name
    return None

def read_nano_gpu_power_once(power_data):
    """
    Returns GPU power in Watts if found, else None.
    Expects power_data from jetson.power (dict with keys 'rail' and 'tot').
    """
    rail = power_data.get("rail")
    if not isinstance(rail, dict):
        return None
    gpu_rail = find_nano_gpu_rail_name(rail)
    if not gpu_rail:
        return None

    entry = rail.get(gpu_rail, {})
    # jtop reports mW for 'power'
    mw = entry.get("power")
    if mw is None:
        return None
    return float(mw) / 1000.0  # W


def log_gpu_stats(
    gpu_index=0,
    log_interval=0.5,
    log_file="gpu_power_log.csv",
    stop_event: Optional[Event] = None,
    shared_power_list=None,
):
    """
    Continuously logs power, utilization, and temperature.
    Appends power readings (in Watts) to shared_power_list if provided.
    Stops when stop_event is set.
    """
    if _NVML_AVAILABLE:
        nvmlInit()
        try:
            device_count = nvmlDeviceGetCount()
            if gpu_index >= device_count:
                logger.error(f"Error: GPU index {gpu_index} out of range (0..{device_count-1})")
                return

            handle = nvmlDeviceGetHandleByIndex(gpu_index)

            with open(log_file, "w") as f:
                f.write("timestamp,gpu_index,power_w,utilization_pct,temperature_c\n")

            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0
                util  = nvmlDeviceGetUtilizationRates(handle).gpu
                temp  = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

                # write to CSV
                with open(log_file, "a") as f:
                    f.write(f"{ts},{gpu_index},{power:.2f},{util},{temp}\n")

                # record in shared list (if exists)
                if shared_power_list is not None:
                    shared_power_list.append(power)

                time.sleep(log_interval)
        finally:
            nvmlShutdown()
    
    # Jetson Nano
    elif _JTOP_AVAILABLE:
        try:
            with jtop() as jetson:
                # Show what rails are available
                rails = list(jetson.power.get("rail", {}).keys())
                logger.info("Connected to Jetson via jtop.")
                if rails:
                    logger.info(f"Available rails: {rails}")
                else:
                    logger.info(f"(none)")

                while jetson.ok():
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    pwr_w = read_nano_gpu_power_once(jetson.power)
                    if pwr_w is not None:
                        # logger.info(f"GPU Power: {pwr_w:.2f} W")
                        # write to CSV
                        with open(log_file, "a") as f:
                            f.write(f"{ts},{gpu_index},{pwr_w:.2f}\n")
                        # record in shared list (if exists)
                        if shared_power_list is not None:
                            shared_power_list.append(pwr_w)
                    else:
                        # Help debug: show rails once in a while
                        pd = jetson.power
                        looger.warning("GPU rail not found. Keys:",
                            {"rail": list(pd.get("rail", {}).keys()),
                            "tot": list(pd.get("tot", {}).keys()) if isinstance(pd.get("tot"), dict) else pd.get("tot")})
                    time.sleep(log_interval)
        finally:
            # process stop
            pass


def launch_energy_logger_process():
    logical_id, physical_id, pci_bus_id = get_real_gpu_index()
    manager = Manager()
    power_list = manager.list()
    stop_event = Event()
    # Launch logger
    proc = Process(
        target=log_gpu_stats,
        kwargs=dict(
            gpu_index=physical_id,
            log_interval=0.1,
            log_file=f"gpu{physical_id}_power_log.csv",
            stop_event=stop_event,
            shared_power_list=power_list,
        ),
    )
    proc.start()
    return stop_event, power_list, proc

def stop_energy_logger_process(proc, stop_event):
    # signal the logger to stop and wait for it
    stop_event.set()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()  # hard-stop as a fallback


def trace_handler(prof: torch.profiler.profile, dir_name="torch_profile_output",
                  worker_name = None, use_gzip: bool = False,
                  file_prefix="prefilling", device="cuda:0"):
    if not os.path.isdir(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            raise RuntimeError("Can't create directory: " + dir_name) from e
    if not worker_name:
        worker_name = f"{socket.gethostname()}_{os.getpid()}"
    # Use nanosecond here to avoid naming clash when exporting the trace
    timestamp = time.time_ns()
    file_name = f"{file_prefix}.{worker_name}.{timestamp}.pt.trace.json"
    if use_gzip:
        file_name = file_name + ".gz"
    prof.export_chrome_trace(os.path.join(dir_name, file_name))
    # Fix the rank issue for  HolisticTraceAnalysis
    # reference: https://github.com/facebookresearch/HolisticTraceAnalysis/issues/107
    # FIXME: This does not work for json.gz
    # rn_rank = np.random.randint(low=0, high=16, dtype=int) # If there are multiple traces files, then each file should have a unique rank value.
    if use_gzip:
        with gzip.open(os.path.join(dir_name, file_name), mode="rt") as fin:
            data = json.loads(fin.read())
        data["distributedInfo"] = {"rank": 0} # must use 0. I don't know why. If there are multiple traces files, then each file should have a unique rank value.
        with gzip.open(os.path.join(dir_name, file_name), 'w') as fout:
            fout.write(json.dumps(data).encode('utf-8')) 
    else:
        with open(os.path.join(dir_name, file_name), "r") as fin:
            data = json.load(fin)
        data["distributedInfo"] = {"rank": 0} # must use 0. I don't know why. If there are multiple traces files, then each file should have a unique rank value.
        with open(os.path.join(dir_name, file_name), "w") as fout:
            json.dump(data, fout, indent=2)

    analyzer = TraceAnalysis(trace_files={0: file_name}, trace_dir=dir_name)
    kernel_type_metrics_df, kernel_metrics_df = analyzer.get_gpu_kernel_breakdown(visualize=False, num_kernels=100)
    kernel_type_metrics_df.to_csv(os.path.join(dir_name, f'kernel_type_metrics.{file_prefix}.{timestamp}.csv'), index=False)
    kernel_metrics_df.to_csv(os.path.join(dir_name, f'kernel_metrics.{file_prefix}.{timestamp}.csv'), index=False)
    # this feature is at https://github.com/facebookresearch/HolisticTraceAnalysis/pull/209
    # To get accurate kernel results, checkout this branch https://github.com/hychiang-git/HolisticTraceAnalysis/tree/dev/no_merge_cpu_kernels
    if hasattr(analyzer, "get_gpu_user_annotation_breakdown"):
        try:
            user_annotation_kernel_type_metrics_df, user_annotation_metrics_df = analyzer.get_gpu_user_annotation_breakdown(visualize=False, num_kernels=100)
            user_annotation_kernel_type_metrics_df.to_csv(os.path.join(dir_name, f'user_annotation_kernel_type_metrics.{file_prefix}.{timestamp}.csv'), index=False)
            user_annotation_metrics_df.to_csv(os.path.join(dir_name, f'user_annotation_metrics.{file_prefix}.{timestamp}.csv'), index=False)
        except Exception as e:
            logger.warning(f"Failed to get user annotation breakdown: {e}")
    # Construct the memory timeline file.
    # !!! This does not work for graph cache !!!
    html_name = f"{file_prefix}.{worker_name}.{timestamp}.html"
    prof.export_memory_timeline(os.path.join(dir_name, html_name), device=device)


def profile_size(model, model_type, batch_size=1, prompt_len=1024, use_GiB=False):

    logger.info(">>> Profiling model size")
    logger.info("Start profiling...")
    # get model total size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    # create model abs
    model_helper = ModelHelperRegistry[model_type](model_config=model.config)
    # model conv/ssm/kv caches
    GB = 1000**3 # This is the SI (base-10) definition used by most storage manufacturers.
    MB = 1000**2
    if use_GiB:
        GB = 1024**3 # Binary (used by Linux/OS), aka GiB
        MB = 1024**2 # Binary (used by Linux/OS), aka MiB
    conv_state_size, ssm_state_size, kv_cache_size = model_helper.get_cache_size(batch_size, prompt_len)
    cache_gb = (conv_state_size + ssm_state_size + kv_cache_size) / GB
    logger.info(f'cache size: {cache_gb:.3f} GB (batch size {batch_size}, prompt length {prompt_len}), detailed breakdown:')
    
    # Only log conv/ssm states if they exist (for Mamba-based models)
    if conv_state_size > 0:
        logger.info(f"-- conv state: {conv_state_size / GB:.3f} GB")
    if ssm_state_size > 0:
        logger.info(f"-- ssm state: {ssm_state_size / GB:.3f} GB")
    logger.info(f"-- kv cache: {kv_cache_size / GB:.3f} GB")
    
    # model total size and detailed layer type breakdown
    model_size = (param_size + buffer_size)
    layer_size_dict = model_helper.get_layer_size(model)
    logger.info(f'model size: {model_size / GB:.3f} GB, detailed breakdown:')
    layer_size_sum = 0
    for k, v in layer_size_dict.items():
        if v < MB:
            logger.info(f"-- {k}: {v / MB:.3f} MB")
        else:
            logger.info(f"-- {k}: {v / GB:.3f} GB")
        layer_size_sum += v
    assert layer_size_sum == model_size, f"Model size breakdown does not match the total model size: {layer_size_sum} != {model_size}"


def estimate_quantized_size(model, model_type, w_bits, batch_size=1, prompt_len=1024, use_GiB=False):

    logger.info(f">>> Estimating quantized w{w_bits}a16 model size")
    logger.info("Start estimating...")

    total_size = 0
    model_helper = ModelHelperRegistry[model_type](model_config=model.config)

    # we always estimate the size of 4-bit quantization
    embeddings = model_helper.get_embeddings(model)
    embed_dtype = embeddings.weight.dtype
    embed_weight = embeddings.weight.data
    scale_size = embed_weight.shape[0] * embed_weight.element_size() # per-token scales * number of bytes
    embed_size = embed_weight.nelement() * 4 / 8 # number of elements * 4 bits per element (/ 8 bits to bytes)
    total_size += scale_size + embed_size

    # estimate the size of layers with w_bits
    layers = model_helper.get_layers(model)
    group_size = 128
    layer_size_sum = 0
    for layer_idx, layer in enumerate(layers):

        # HOTFIX: for mamba layer
        if hasattr(layer, "A_log"):
            layer_size_sum += module.A_log.nelement() * module.A_log.element_size()
        if hasattr(layer, "dt_bias"):
            layer_size_sum += module.dt_bias.nelement() * module.dt_bias.element_size()
        if hasattr(layer, "D"):
            layer_size_sum += module.D.nelement() * module.D.element_size()
        
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                logger.warning(f"Estimating quantized layer.{layer_idx}.{name} with {w_bits} bits")
                g_size = group_size if module.in_features > group_size else -1
                out_features, in_features = module.weight.shape
                num_groups = math.ceil(in_features / g_size) # we will pad in_feature (i.e., k)
                pad_out = 0
                if out_features % 256 != 0:
                    pad_out = 256 - out_features % 256
                scale_size = num_groups * (out_features + pad_out) * module.weight.element_size()
                weight_size = in_features * (out_features + pad_out) * w_bits / 8 # number of elements * w_bits per element (/ 8 bits to bytes)
                layer_size_sum += scale_size + weight_size
            elif ("norm" in name) or ("conv1d" in name):
                param_size = 0
                for param in module.parameters():
                    param_size += param.nelement() * param.element_size()
                buffer_size = 0
                for buffer in module.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
                layer_size_sum += param_size + buffer_size
            else:
                logger.debug(f"Skipping layer.{layer_idx}.{name}")
    total_size += layer_size_sum

    # estimate the size of lm_head
    head = model_helper.get_lm_head(model)
    out_features, in_features = head.weight.shape
    g_size = group_size if in_features > group_size else -1
    num_groups = math.ceil(in_features / g_size) # we will pad in_feature (i.e., k)
    pad_out = 0
    if out_features % 256 != 0:
        pad_out = 256 - out_features % 256
    scale_size = num_groups * (out_features + pad_out) * head.weight.element_size()
    weight_size = in_features * (out_features + pad_out) * w_bits / 8 # number of elements * w_bits per element (/ 8 bits to bytes)
    total_size += scale_size + weight_size


    # model conv/ssm/kv caches
    GB = 1000**3 # This is the SI (base-10) definition used by most storage manufacturers.
    MB = 1000**2
    if use_GiB:
        GB = 1024**3 # Binary (used by Linux/OS), aka GiB
        MB = 1024**2 # Binary (used by Linux/OS), aka MiB

    conv_state_size, ssm_state_size, kv_cache_size = model_helper.get_cache_size(batch_size, prompt_len)
    cache_gb = (conv_state_size + ssm_state_size + kv_cache_size) / GB
    logger.info(f'cache size: {cache_gb:.3f} GB (batch size {batch_size}, prompt length {prompt_len}), detailed breakdown:')
    
    # Only log conv/ssm states if they exist (for Mamba-based models)
    if conv_state_size > 0:
        logger.info(f"-- conv state: {conv_state_size / GB:.3f} GB")
    if ssm_state_size > 0:
        logger.info(f"-- ssm state: {ssm_state_size / GB:.3f} GB")
    logger.info(f"-- kv cache: {kv_cache_size / GB:.3f} GB")

    # get model total size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size = (param_size + buffer_size)
    logger.info(f"Actual model size: {model_size / GB:.3f} GB")
    logger.info(f"Estimated quantized model size: {total_size / GB:.3f} GB")


def profile_ttft(model, batch_size=1, prompt_len=1024, repeats=100, torch_profile=False, torch_profile_dir=""):
    # no graph cache mode for TTFT (prefilling stage)
    logger.info(f">>> Profiling TTFT (prefilling stage) for {repeats} times")
    # For Llama2, use vocabulary size from model config
    vocab_size = model.config.vocab_size
    prompt = torch.randint(low=0, high=vocab_size, size=(batch_size, prompt_len,)).to(torch.cuda.current_device())
    
    logger.info(f"Testing (batch_size, prompt_len): ({batch_size}, {prompt_len})")
    logger.info("Warmup...")
    with torch.no_grad():
        for _ in range(5):
            # For Llama2, use standard forward pass with use_cache=True
            _ = model(prompt, use_cache=True, output_hidden_states=False, output_attentions=False)
    torch.cuda.synchronize()

    logger.info("Start profiling...")
    if args.energy:
        logger.info("Launch energy logger process...")
        stop_event, power_list, proc = launch_energy_logger_process()

    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True) 
        end   = torch.cuda.Event(enable_timing=True) 
        start.record()
        for _ in range(repeats):
            _ = model(prompt, use_cache=True, output_hidden_states=False, output_attentions=False)
        end.record()
        torch.cuda.synchronize()
    dur = start.elapsed_time(end)
    logger.info(f"Finished, latency: {dur/repeats:.2f} milliseconds")

    if args.energy:
        stop_energy_logger_process(proc, stop_event)
        watts = list(power_list)
        n_trunc = len(watts) // 10
        watts = watts[n_trunc:-n_trunc] # truncated head and tails
        avg_power = sum(watts) / len(watts) if watts else 0.0
        energy_joules = (avg_power * (dur / 1000.0)) / repeats  # approximate energy in Joules
        logger.info(f"Collected power {len(watts)} samples, avg power = {avg_power:.2f} W, energy = {energy_joules:.2f} J / prompt")

    if torch_profile:
        logger.info("Run torch profiler...")
        outfile_prefix = f"ttft_prompt_len_{prompt_len}"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler, dir_name=torch_profile_dir, use_gzip=True, file_prefix=outfile_prefix, device="cuda:0"
            )
        ) as prof:

            with torch.no_grad():
                # (wait=0, warmup=0, active=5) , repeat=1
                for _ in range(5):
                    with record_function("## forward ##"):
                        out = model(prompt, use_cache=True, output_hidden_states=False, output_attentions=False)
                    prof.step()


def profile_tpot(model, batch_size=1, prompt_len=1024, repeats=100, cache_graph=False, torch_profile=False, torch_profile_dir=""):
    logger.info(f">>> Profiling TPOT (generation stage) for {repeats} times, cache_graph: {cache_graph}")
    
    # For Llama2, use vocabulary size from model config
    vocab_size = model.config.vocab_size
    device = next(iter(model.parameters())).device
    
    # Create a dummy prompt for the prefilling stage (to populate KV cache)
    dummy_prompt = torch.randint(low=0, high=vocab_size, size=(batch_size, prompt_len,)).to(device)
    
    # Prefill the KV cache
    logger.info("Prefilling KV cache...")
    with torch.no_grad():
        # Run prefilling to populate KV cache
        outputs = model(dummy_prompt, use_cache=True, output_hidden_states=False, output_attentions=False)
        if hasattr(outputs, "past_key_values"):
            past_key_values = outputs.past_key_values
        else:
            # Nemotron-H and Mamba2
            past_key_values = outputs.cache_params
    
    # Single token input for generation
    input_token = torch.randint(low=0, high=vocab_size, size=(batch_size, 1)).to(device)
    cache_position = torch.arange(1, device=dummy_prompt.device)
    
    # Warmup
    logger.info("Warmup...")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad():
        with torch.cuda.stream(s):
            for _ in range(5):
                # Use past_key_values for efficient generation
                # to compatible with Nemotron-H and Mamba2, we pass cache_params here
                _ = model(input_token, past_key_values=past_key_values, cache_params=past_key_values,
                          cache_position=cache_position, use_cache=True, output_hidden_states=False, output_attentions=False)
    torch.cuda.current_stream().wait_stream(s)
    
    if cache_graph:
        with torch.no_grad():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # to compatible with Nemotron-H and Mamba2, we pass cache_params here
                out = model(input_token, past_key_values=past_key_values, cache_params=past_key_values,
                            cache_position=cache_position, use_cache=True, output_hidden_states=False, output_attentions=False)
            
        def generate(new_input_token, new_past_key_values):
            input_token.copy_(new_input_token)
            # Note: CUDA graph replay with dynamic past_key_values is complex
            # This is a simplified version - in practice, you might need more sophisticated handling
            graph.replay()
            return out
    else:
        def generate(new_input_token, new_past_key_values):
            # to compatible with Nemotron-H and Mamba2, we pass cache_params here
            out = model(new_input_token, past_key_values=new_past_key_values, cache_params=new_past_key_values,
                        cache_position=cache_position, use_cache=True, output_hidden_states=False, output_attentions=False)
            return out
        
    logger.info("Start profiling...")
    if args.energy:
        logger.info("Launch energy logger process...")
        stop_event, power_list, proc = launch_energy_logger_process()

    new_input_token = torch.randint(low=0, high=vocab_size, size=(batch_size, 1)).to(device)
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True) 
        end   = torch.cuda.Event(enable_timing=True) 
        start.record()
        for _ in range(repeats):
            generate(new_input_token, past_key_values)
        end.record()
        torch.cuda.synchronize()
    dur = start.elapsed_time(end)
    logger.info(f"Finished, latency: {dur/repeats:.2f} milliseconds (cache_graph={cache_graph})")

    if args.energy:
        stop_energy_logger_process(proc, stop_event)
        watts = list(power_list)
        avg_power = sum(watts) / len(watts) if watts else 0.0
        energy_joules = (avg_power * (dur / 1000.0)) / repeats  # approximate energy in Joules
        logger.info(f"Collected power {len(watts)} samples, avg power = {avg_power:.2f} W, energy = {energy_joules:.2f} J / token")

    if torch_profile:
        logger.info("Run torch profiler...")
        outfile_prefix = f"tpot"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler, dir_name=torch_profile_dir, use_gzip=False, file_prefix=outfile_prefix, device="cuda:0"
            )
        ) as prof:

            with torch.no_grad():
                # (wait=0, warmup=0, active=5) , repeat=1
                for _ in range(5):
                    generate(new_input_token, past_key_values)
                    prof.step()



def profile_ttlt(model, batch_size=1, prompt_len=1024, gen_len=128, repeats=100, cache_graph=False, torch_profile=False, torch_profile_dir=""):
    logger.info(f">>> Profiling TTLT (prefilling + generation) for {repeats} times, cache_graph: {cache_graph}")
    logger.info(f"batch_size: {batch_size}, prompt_len: {prompt_len}, gen_len:{gen_len}")

    # For Llama2, use vocabulary size from model config
    vocab_size = model.config.vocab_size
    device = next(iter(model.parameters())).device
    max_length = prompt_len + gen_len

    # cache the graph for generation
    if cache_graph:
        # Create a dummy prompt for the prefilling stage (to populate KV cache)
        dummy_prompt = torch.randint(low=0, high=vocab_size, size=(batch_size, prompt_len,)).to(device)
        
        # Prefill the KV cache
        with torch.no_grad():
            outputs = model(dummy_prompt, use_cache=True, output_hidden_states=False, output_attentions=False)
            # Nemotron-H and Mamba2
            if hasattr(outputs, "past_key_values"):
                past_key_values = outputs.past_key_values
            else:
                past_key_values = outputs.cache_params
        
        input_token = torch.randint(low=0, high=vocab_size, size=(batch_size, 1)).to(device)
        cache_position = torch.arange(1, device=dummy_prompt.device)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad():
            with torch.cuda.stream(s):
                for _ in range(10):
                    # to compatible with Nemotron-H and Mamba2, we pass cache_params here
                    out = model(input_token, past_key_values=past_key_values, cache_params=past_key_values,
                                cache_position=cache_position, use_cache=True, output_hidden_states=False, output_attentions=False)
        torch.cuda.current_stream().wait_stream(s)

        with torch.no_grad():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                    # to compatible with Nemotron-H and Mamba2, we pass cache_params here
                out = model(input_token, past_key_values=past_key_values, cache_params=past_key_values,
                            cache_position=cache_position, use_cache=True, output_hidden_states=False, output_attentions=False)
        
        def generate(new_input_token, new_past_key_values):
            input_token.copy_(new_input_token)
            # Note: CUDA graph replay with dynamic past_key_values is complex
            # This is a simplified version - in practice, you might need more sophisticated handling
            graph.replay()
            return out
    else:
        def generate(new_input_token, new_past_key_values):
            # to compatible with Nemotron-H and Mamba2, we pass cache_params here
            out = model(new_input_token, past_key_values=new_past_key_values, cache_params=new_past_key_values,
                        cache_position=cache_position, use_cache=True, output_hidden_states=False, output_attentions=False)
            return out

    def run(batch_size, prompt_len, gen_len):
        max_length = prompt_len + gen_len
        prompt = torch.randint(low=0, high=vocab_size, size=(batch_size, prompt_len)).to(device)
        sequences = [prompt]
        
        # prefilling
        outputs = model(sequences[-1], use_cache=True, output_hidden_states=False, output_attentions=False)
        # Nemotron-H and Mamba2
        if hasattr(outputs, "past_key_values"):
            past_key_values = outputs.past_key_values
        else:
            past_key_values = outputs.cache_params
        sampled_tokens = outputs.logits[:, -1, :].argmax(dim=-1)  # Get last token logits
        sampled_tokens = sampled_tokens.unsqueeze(1)  # "b -> b 1"
        sequences.append(sampled_tokens)
        
        # generate
        current_past_key_values = past_key_values
        for _ in range(gen_len - 1):  # -1 because we already generated one token
            outputs = generate(sequences[-1], current_past_key_values)
            if hasattr(outputs, "past_key_values"):
                current_past_key_values = outputs.past_key_values
            else:
                current_past_key_values = outputs.cache_params
            sampled_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
            sampled_tokens = sampled_tokens.unsqueeze(1)  # "b -> b 1"
            sequences.append(sampled_tokens)

    logger.info("Warmup...")
    with torch.no_grad():
        for _ in range(5):
            run(batch_size, prompt_len, gen_len)

    logger.info("Start profiling...")
    if args.energy:
        logger.info("Launch energy logger process...")
        stop_event, power_list, proc = launch_energy_logger_process()

    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True) 
        end   = torch.cuda.Event(enable_timing=True) 
        start.record()
        for _ in range(repeats):
            run(batch_size, prompt_len, gen_len)
        end.record()
        torch.cuda.synchronize()
    dur = start.elapsed_time(end)
    logger.info(f"Finished, latency: {dur/repeats:.2f} milliseconds (cache_graph={cache_graph})")
    
    if args.energy:
        stop_energy_logger_process(proc, stop_event)
        watts = list(power_list)
        avg_power = sum(watts) / len(watts) if watts else 0.0
        energy_joules = (avg_power * (dur / 1000.0)) / repeats  # approximate energy in Joules
        logger.info(f"Collected power {len(watts)} samples, avg power = {avg_power:.2f} W, energy = {energy_joules:.2f} J / request")
    
    if torch_profile:
        logger.info("Run torch profiler...")
        logger.warning("Profile ttlt with torch profiler is very slow...")
        outfile_prefix = f"ttlt_prompt_len_{prompt_len}_gen_len_{gen_len}_cache_graph_{cache_graph}"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler, dir_name=torch_profile_dir, use_gzip=False, file_prefix=outfile_prefix, device="cuda:0"
            )
        ) as prof:

            with torch.no_grad():
                # (wait=0, warmup=0, active=5) , repeat=1
                for _ in range(5):
                    run(batch_size, prompt_len, gen_len)
                    prof.step()

                    
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

    if args.size:
        if contains_wxa16(args.model_repo):
            w_bits = get_wxa16_number(args.model_repo)
            if w_bits == 4:
                # profile w4a16 model size
                profile_size(model, model_type, args.batch_size, args.prompt_len)
            else:
                # estimate model size
                logger.warning(f"The size of w{w_bits}a16 models is estimated.")
                estimate_quantized_size(model, model_type, w_bits, args.batch_size, args.prompt_len)
        else:
            # profile fp16 model size
            profile_size(model, model_type, args.batch_size, args.prompt_len)

    if args.ttft:
        profile_size(model, model_type, args.batch_size, args.prompt_len)
        if args.cache_graph:
            logger.warning("TTFT does not support cache_graph mode, ignore cache_graph")
        profile_ttft(model, args.batch_size, args.prompt_len, args.repeats, args.torch_profile, f"torch_profile/{model_name}")

    if args.tpot:
        if args.gen_len > 1:
            logger.warning("TPOT only test the latency with the given prompt length, ignore gen_len")
        profile_size(model, model_type, args.batch_size, args.prompt_len)
        profile_tpot(model, args.batch_size, args.prompt_len, args.repeats, args.cache_graph, args.torch_profile, f"torch_profile/{model_name}")

    if args.ttlt:
        profile_size(model, model_type, args.batch_size, args.prompt_len)
        profile_ttlt(model, args.batch_size, args.prompt_len, args.gen_len, args.repeats, args.cache_graph, args.torch_profile, f"torch_profile/{model_name}")

    if not args.size and not args.ttft and not args.tpot and not args.ttlt:
        logger.warning("No profiling task to run with, try `--ttft`, `--tpot`, `--ttlt`, `--size`?")

if __name__ =='__main__':    
    # Fix all possible random seef for reproduce
    set_deterministic(1234)
    torch.backends.cudnn.benchmark = True
    # parse args
    parser = argparse_shared_options(argparse.ArgumentParser())
    parser = argparse_compress_options(parser)
    parser.add_argument(
        '--cache_graph', action='store_true', default=False,
        help='To enable CUDA graph cache, this only works for the generation stage (TPOT and TTLT)'
    )
    parser.add_argument(
        '--repeats', type=int, default=100,
        help='The number of profiling to repeat (default: 100)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='The input batch size. (default: 1)'
    )
    parser.add_argument(
        '--prompt_len', type=int, default=1024,
        help='The number of input tokens. (default: 1024)'
    )
    parser.add_argument(
        '--gen_len', type=int, default=128,
        help='The number of generation tokens output. This is only for TTLT. (default: 128)'
    )
    parser.add_argument(
        '--size', action='store_true',
        help='Profile model total size (i.e. parameters + buffers)'
    )
    parser.add_argument(
        '--ttft', action='store_true',
        help='Profile time to first token (TTFT, i.e. prefilling stage)'
    )
    parser.add_argument(
        '--tpot', action='store_true',
        help='Profile time per output token (TPOT, i.e. generation stage)'
    )
    parser.add_argument(
        '--ttlt', action='store_true',
        help='Profile time to generate a sequence (TTLT, i.e. prefilling + generation)'
    )
    parser.add_argument(
        '--torch_profile', action='store_true',
        help='Whether to launch the pytorch profiler.'
    )
    parser.add_argument(
        '--energy', action='store_true',
        help='Whether to profile the GPU power and energy.'
    )
    args = parser.parse_args()
    if args.verbose:
        set_logger(logger, logging.DEBUG)
    else:
        set_logger(logger, logging.INFO)
    main(args)
