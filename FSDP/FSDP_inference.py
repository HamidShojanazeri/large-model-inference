import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM


import os
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# for grammar correction
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# for generation
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq

import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)


from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)


from policies import mixed_precision

from datasets import load_dataset, load_metric
# from torch.utils.data import DataLoader
# from pathlib import Path
# from torch.utils.data import DataLoader
# import performance

# from ChildTuningOptimizer import ChildTuningAdamW

# from sklearn.model_selection import train_test_split
import time
from datetime import datetime

# local imports
import verify
import policies

# import datasets_grammar as dg
# import tqdm
# import numpy as np
# from statistics import stdev
# config
import config

# import model_checkpoints
from collections import deque



# some globals
g_gigabyte = 1024**3


def _is_rank_0():
    return 0 == os.getenv("RANK")


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch fsdp T5.11 Example")
    """parser.add_argument("--save-dir", default="/model_chkpt", type=str)
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    """

    args = parser.parse_args()
    return args


# ----------------   Main functions --------------------
def get_policies(cfg, fsdp_unit_params=1000000):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.use_mixed_precision:
        bf16_ready = verify.bf16_ready

        if bf16_ready:
            mixed_precision_policy = policies.bfSixteen
            print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print(f"bFloat16 support not present. Not using for mixed precision")

    wrapping_policy = policies.get_t5_wrapper()

    return mixed_precision_policy, wrapping_policy


def setup(rank, world_size, cfg):
    # os.environ["MASTER_ADDR"] = g_addr
    # os.environ["MASTER_PORT"] = cfg.host_port

    # initialize the process group
    dist.init_process_group("nccl")  # , rank=rank, world_size=world_size)


def setup_environ_flags(cfg, rank):
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    if cfg.nccl_debug_handler:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if cfg.distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if rank == 0:
            print(f"--> running with torch dist debug set to detail")


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    print(f"clearing cache for rank {rank}")
    torch.cuda.empty_cache()


def setup_tasks(rank, world_size, cfg):
    """keep the basic setup list here"""
    setup(rank, world_size, cfg)
    # clear_gpu_cache() - need to call torch set device first?
    # set_printing()
    setup_environ_flags(cfg, rank)


def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num

def format_stats(item, rounding=8):
    return round(item, ndigits=rounding)

# def model_info(in_model):
#     print(f"--> Model T5 1.1")
#     total_params = sum(p.numel() for p in in_model.parameters() if p.requires_grad)
#     print(f"\n--> {model_name[7:]} has {round(total_params/1e9,5)} Billion params\n")

# def inference(input_text):
#     """ takes an input sentence"""
#     input_tokens = tokenizer("grammar:"+input_text, truncation=True, return_tensors="pt")
#     input_tokens.to("cuda:0")
#     output = model.generate(input_tokens['input_ids'], num_beams=5, max_length=512, early_stopping=True)
#     correction=tokenizer.batch_decode(output, skip_special_tokens=True)
#     res = "".join(correction)
#     print(f"Input:  {input_text}")
#     print(f"\nCorrected:  {res}")


def fsdp_inference(args):
    cfg = config.train_config()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup_tasks(rank, world_size, cfg)

    fsdp_unit_params = cfg.fsdp_unit_size

    model_name = "google/t5-v1_1-xl"
    tokenizer_name = "t5-large"


    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=512)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    mp_policy, wrapping_policy = get_policies(cfg, fsdp_unit_params)

    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mp_policy,
        #sharding_strategy=model_sharding_strategy,
        #backward_prefetch=backward_policy,
        device_id=torch.cuda.current_device(),  # streaming init
        #limit_all_gathers=cfg.use_rate_limiter,
        # inflight_max=cfg.inflight_max,
    )

    text = "I is reading about AI articles "
    inputs = tokenizer("grammar:"+text, truncation=True, return_tensors='pt')

    output = model.generate(inputs['input_ids'], num_beams=5, max_length=512, early_stopping=True)
    correction=tokenizer.batch_decode(output, skip_special_tokens=True)
    print("".join(correction))
    dist.barrier()

if __name__ == "__main__":

    args = parse_args()

    gpus_per_node = torch.cuda.device_count()

    # torch run start
    fsdp_inference(args)

# torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="localhost:5679" FSDP_inference.py
