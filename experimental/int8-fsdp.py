import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import policies
from policies import mixed_precision
import config
import verify
import os
import argparse
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

import torch.distributed as dist

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import torch.multiprocessing as mp

gigabyte_size = 1073741824

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def format_to_gb(item, precision=4):
    """quick function to format numbers to gigabyte and round to (default) 4 digit precision"""
    metric_num = item / gigabyte_size
    metric_num = round(metric_num, ndigits=precision)
    return metric_num

def get_policies(cfg):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.use_mixed_precision:
        bf16_ready = verify.bf16_ready

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = policies.bfSixteen
            # if rank == 0:
            #     print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = policies.fpSixteen
            # if rank == 0:
            #     print(f"FP16 enabled. ")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print(
                f"bFloat16 support not present. Will use FP32, and not mixed precision"
            )

    # wrapping policy -------
    # print(f"**overriding mp to fp16 - remove")
    # mixed_precision_policy = policies.fpSixteen

    wrapping_policy = policies.get_t5_wrapper()

    return mixed_precision_policy, wrapping_policy
def inference (rank, WORLD_SIZE, args):
    cfg = config.benchmark_config()

    mp_policy, wrapping_policy = get_policies(cfg)

    MAX_NEW_TOKENS = 10
    model_name = 'facebook/opt-1.3b'
    # model_name = "bigscience/bloom-3b"

    text = """
    Q: On average Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes.
    How many punches did he throw?\n
    A: Let’s think step by step.\n"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-2}GB'

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    start_loading = time.time()
    setup(rank , WORLD_SIZE)
    torch.cuda.empty_cache()

    model= AutoModelForCausalLM.from_pretrained(
    model_name)
    # low_cpu_mem_usage=True)
    #   device_map='auto',
    #   load_in_8bit=True,
    #   max_memory=max_memory

    model = FSDP(
            model,
            auto_wrap_policy=wrapping_policy,
            mixed_precision=mp_policy,
            device_id=torch.cuda.current_device(),
        )

    # model_native = AutoModelForCausalLM.from_pretrained(
    #   model_name,
    #   device_map='auto',
    #   max_memory=max_memory,
    #   torch_dtype="auto"
    # )

    end_loading = time.time()
    print( " model load time is  {} s ".format((end_loading-start_loading)) )

    ##### memory foot print on all devices

    devices_list = torch.cuda.device_count()

    for device in range(devices_list):
        gpu = "cuda:"+ str(device)
        print( f" memory reserved on device {gpu} is {torch.cuda.mem_get_info(device=gpu)}")

    inference_latency = []

    for i in range(10):
     start_inference = time.time()
    input_ids = input_ids.to('cuda')
    generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
    # generated_ids = model_native.generate(input_ids, max_length=MAX_NEW_TOKENS)
    stop_inference = time.time()
    inference_latency.append(stop_inference-start_inference)

    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    avg_inference_latency = sum(inference_latency)/len(inference_latency)
    print( " Avg time for inference is {}s ".format((avg_inference_latency)))

# mem_fp16 = model_native.get_memory_footprint()
# mem_int8 = model_int8.get_memory_footprint()
# print("int8 model memory foot print is {}".format(format_to_gb(mem_int8)))
# print("Memory footprint int8 model: {} | Memory footprint fp16 model: {} | Relative difference: {}".format(mem_int8, mem_fp16, mem_fp16/mem_int8))

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FSDP inference example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for inference (default: 1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(inference,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
