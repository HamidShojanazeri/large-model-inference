import argparse
import logging
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def numerical_test(tensors1, tensors2, rtol, atol):
    """
    truth_tensors is the source of truth.
    test_dict looks like
    [
        (name, out_tensors, atol, rtol),
        ...
    ]
    """
    assert len(tensors1) == len(tensors2)
    n_failures = 0
    max_diff = 0
    for tensor1, tensor2 in zip(tensors1, tensors2):
        # print(tensor1, tensor2)
        max_diff = max(max_diff, torch.max(torch.abs(tensor1 - tensor2)))
        if not torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
            n_failures += 1

    num_tested = len(tensors1)
    return n_failures, max_diff, num_tested


def benchmark_cuda_event(model, inputs, num_batch):
    """
    Benchmark Pippy with cuda events
    """

    iters = len(inputs)
    print("##########",inputs[0]["input_ids"].size(), inputs[0].keys())
    model(**inputs[0])
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_batch):
        model(**inputs[i])
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_batch

def benchmark_time_perfcounter(model, inputs, num_batch):
    """
    Benchmark Pippy with time.perf_counter
    """

    for i in range(num_batch):
        start_event = time.perf_counter()
        model(**inputs[i])
        end_event = time.perf_counter()
    return (end_event - start_event) / num_batch


def get_enc_dec_batch(batch_size, avg_seqlen, max_sequence_length, seqlen_stdev, vocab_size, device, pad_idx=0):
    mean_tensor = torch.Tensor([avg_seqlen]).expand(batch_size)
    stdev_tensor = torch.Tensor([seqlen_stdev]).expand(batch_size)
    lengths = torch.normal(mean_tensor, stdev_tensor).to(torch.int)
    lengths = torch.clamp(lengths, min=0, max=max_sequence_length)
    tokens = torch.full(
        (batch_size, max_sequence_length),
        pad_idx,
    )
    for i in range(batch_size):
        tokens[i, : lengths[i]] = torch.randint(
            pad_idx + 1,
            vocab_size - 1,
            size=(lengths[i],),
        )
    return  {'input_ids': tokens.to(device), 'decoder_input_ids': tokens.to(device)}



def setup_logger(filename="log.csv"):

    # create log file
    if os.path.exists(filename):
        os.remove(filename)
        open(filename, "w").close()
    else:
        open(filename, "w").close()
    # create logger
    lgr = logging.getLogger("logger name")
    lgr.setLevel(logging.DEBUG)  # log all escalated at and above DEBUG
    # add a file handler
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)  # ensure all messages are logged to file

    # create a formatter and set the formatter for the handler.
    # frmt = logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s')
    frmt = logging.Formatter("%(message)s")
    fh.setFormatter(frmt)

    # add the Handler to the logger
    lgr.addHandler(fh)
    return lgr
