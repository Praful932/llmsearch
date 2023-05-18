"""
    Inspired from toma - https://github.com/BlackHC/toma
"""

import gc
import torch

def batch(func):
    """Perform Inference on a batch of samples by dividing the batch_size by 2
    to avoid OOM error

    Args:
        func (_type_): _description_
    """
    def inner_wrapper(*args, batch_size, **kwargs):
        while True:
            try:
                res = func(*args,batch_size = batch_size, **kwargs)
                return res
            except RuntimeError as exception:
                if batch_size > 1 and should_reduce_batch_size(exception):
                    batch_size //= 2
                    gc_cuda()
                else:
                    raise
    return inner_wrapper

def gc_cuda():
    """Gargage collect Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def should_reduce_batch_size(exception):
    return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception) or is_out_of_cpu_memory(exception)

def is_cuda_out_of_memory(exception):
    return (
        isinstance(exception, RuntimeError) and len(exception.args) == 1 and "CUDA out of memory." in exception.args[0]
    )

def is_cudnn_snafu(exception):
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )

def is_out_of_cpu_memory(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )