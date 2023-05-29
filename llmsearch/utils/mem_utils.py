"""
    Inspired from toma - https://github.com/BlackHC/toma
"""

import gc
import inspect
import traceback
import torch

import pynvml
import math
import psutil

class Cache:
    def __init__(self):
        self.cache = {}

    def set_value(self,value, stacktrace, total_available_gpu_memory, total_available_ram_memory):
        hash_key = (stacktrace, total_available_gpu_memory, total_available_ram_memory)
        self.cache[hash_key] = value

    def get_value(self, initial_value, stacktrace, total_available_gpu_memory, total_available_ram_memory):
        hash_key = (stacktrace, total_available_gpu_memory, total_available_ram_memory)
        if hash_key in self.cache:
            val = self.cache[hash_key]
            return val
        return initial_value

    def is_empty(self):
        return not bool(len(self.cache))

    def empty_cache(self):
        self.cache = {}

def get_traceback(ignore_first = 0, stack_context = 5):
    stack = inspect.stack(context=1)[ignore_first:ignore_first+stack_context]
    simple_traceback = tuple(
        (fi.function, fi.code_context[0]) for fi in stack
    )
    return simple_traceback


cache = Cache()

def batch(func):
    """Perform Inference on a batch of samples by dividing the batch_size by 2
    to avoid OOM error

    Args:
        func (_type_): _description_
    """
    def inner_wrapper(*args, batch_size, disable_batch_size_cache, **kwargs):

        """
        cache will only help in the inital initalization - something better than random intialization
        1. intially Get cached batch size if available else, return the batch size that was passed in
            hash function for cache
            traceback
            gpu memory
            ram memory
        2.
        """

        if disable_batch_size_cache:
            # Empty cache
            if not cache.is_empty():
                cache.empty_cache()
        else:
            if not cache.is_empty():
                # Get cached batch size if present
                stacktrace = get_traceback(ignore_first=20, stack_context=10)
                total_available_gpu_memory = get_gpu_information()[2]
                total_available_ram_memory = get_total_available_ram()
                batch_size = cache.get_value(initial_value=batch_size, stacktrace=stacktrace,total_available_gpu_memory=total_available_gpu_memory, total_available_ram_memory=total_available_ram_memory)
        while True:
            try:
                res = func(*args,batch_size = batch_size,disable_batch_size_cache=disable_batch_size_cache, **kwargs)
                gc_cuda()
                if not disable_batch_size_cache:
                    stacktrace = get_traceback(ignore_first=20, stack_context=10)
                    total_available_gpu_memory = get_gpu_information()[2]
                    total_available_ram_memory = get_total_available_ram()
                    print(f"Total available ram memory - {total_available_ram_memory}, Total available gpu memory - {total_available_gpu_memory}\n")
                    # Set value for next iteration with the input hash
                    cache.set_value(value=batch_size, stacktrace=stacktrace,total_available_gpu_memory=total_available_gpu_memory, total_available_ram_memory=total_available_ram_memory)
                return res
            except RuntimeError as exception:
                if batch_size > 1 and should_reduce_batch_size(exception):
                    print(f"Unable to fit batch size - {batch_size}, Reducing batch size to - {batch_size // 2}")
                    batch_size //= 2
                    gc_cuda()
                else:
                    print("Unable to fit the lowest batch size")
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

def get_gpu_information():
    pynvml.nvmlInit()
    try:
        num_gpus = pynvml.nvmlDeviceGetCount()

        total_available_gpus = 0
        total_occupied_memory = 0
        total_memory = 0

        for index in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            total_available_gpus += 1
            total_occupied_memory += memory_info.used
            total_memory += memory_info.total

        # Round occupied memory to the nearest GB
        total_occupied_memory_gb = math.ceil(total_occupied_memory / (1024**3))

        # Round total memory to the nearest GB
        total_memory_gb = math.ceil(total_memory / (1024**3))

        total_available_memory = total_memory_gb - total_occupied_memory_gb

        return total_available_gpus, total_occupied_memory_gb, total_available_memory
    except Exception:
        exc_traceback = traceback.format_exc()
        print(f"Unable to get details of gpu - {exc_traceback}")
        raise
    finally:
        pynvml.nvmlShutdown()

def get_total_available_ram():
    memory = psutil.virtual_memory()
    total_available_ram = memory.available

    # Round available RAM to the nearest GB
    total_available_ram_gb = math.ceil(total_available_ram / (1024**3))

    return total_available_ram_gb