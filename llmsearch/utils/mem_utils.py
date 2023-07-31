"""
Inspired from toma - https://github.com/BlackHC/
Memory related utils to do friendly inference :)
"""

import gc
import math
import inspect
import traceback

from functools import wraps
from typing import Any, Union, Tuple

import torch
import psutil
import pynvml


class Cache:
    """Cache to store the optimal batch size for a specific configuration call"""

    def __init__(self):
        """Initializes cache"""
        self.cache = {}

    def set_value(
        self,
        value: Any,
        stacktrace: Tuple,
        total_available_gpu_memory: float,
        total_available_ram_memory: float,
    ):
        """Sets a value based on certain combination of different hashes

        Args:
            value (Any): hash value
            stacktrace (Tuple): stacktrace of the method call
            total_available_gpu_memory (float): total available gpu memory
            total_available_ram_memory (float): total available ram memory
        """
        # Create hash key
        hash_key = (stacktrace, total_available_gpu_memory, total_available_ram_memory)
        # Set value
        self.cache[hash_key] = value

    def get_value(
        self,
        current_value: Any,
        stacktrace: Tuple,
        total_available_gpu_memory: float,
        total_available_ram_memory: float,
    ) -> float:
        """Tries to get a value for a particular set of hashes, returns `current_value` (this only happens during the initial call)

        Args:
            current_value (Any): current value
            stacktrace (Tuple): stacktrace of the method call
            total_available_gpu_memory (float): total available gpu memory
            total_available_ram_memory (float): total available ram memory

        Returns:
            float: `current_value` if hash_key is not present, else returns the hashed key
        """
        hash_key = (stacktrace, total_available_gpu_memory, total_available_ram_memory)
        # Checks if current configuration in cache
        if hash_key in self.cache:
            val = self.cache[hash_key]
            return val
        return current_value

    def is_empty(self) -> bool:
        """Checks if the cache is empty

        Returns:
            bool: empty or not
        """
        return not bool(len(self.cache))

    def empty_cache(self):
        """Empties cache"""
        self.cache = {}


def get_traceback(ignore_first: int = 0, stack_context: int = 5) -> Tuple[Tuple]:
    """Get traceback from first to latest call

    Args:
        ignore_first (int, optional): ignore first n traceback. Defaults to 0.
        stack_context (int, optional): context for traceback. Defaults to 5.

    Returns:
        Tuple[Tuple]: Tuples of Function call and code
    """
    stack = inspect.stack(context=1)[ignore_first : ignore_first + stack_context]
    # Tuple of function and calling code
    simple_traceback = tuple((fi.function, fi.code_context[0]) for fi in stack)
    return simple_traceback


# init cache
cache = Cache()


def batch_without_oom_error(func : callable):
    """Perform Inference on a batch of samples by dividing the batch_size by 2 each time whenever OOM error happens
    - Function should have a `batch_size` and `disable_batch_size_cache` parameter

    Args:
        func (Callable): function having this signature of arguments `*args, batch_size, disable_batch_size_cache, **kwargs`
    """

    @wraps(func)
    def inner_wrapper(*args, batch_size: int, disable_batch_size_cache: bool, **kwargs):
        """When cache is turned on (`disable_batch_size_cache` - True), a cached batch_size can be used based on the configuration

        Args:
            batch_size (int): initial batch size
            disable_batch_size_cache (bool): setting this to `True` forces the inference to happen using `batch_size`

        """

        # if batch size cache is to be disabled
        if disable_batch_size_cache:
            # Empty cache
            if not cache.is_empty():
                cache.empty_cache()
        else:
            if not cache.is_empty():
                stacktrace = get_traceback(ignore_first=20, stack_context=10)
                gpu_info = get_gpu_information()
                total_available_gpu_memory = gpu_info[2] if gpu_info else gpu_info
                total_available_ram_memory = get_total_available_ram()
                # Get cached batch size if present
                batch_size = cache.get_value(
                    current_value=batch_size,
                    stacktrace=stacktrace,
                    total_available_gpu_memory=total_available_gpu_memory,
                    total_available_ram_memory=total_available_ram_memory,
                )
        while True:
            # Try running with specified batch size
            try:
                # TODO : add logging
                print(f"Performing inference with batch_size - {batch_size}")
                res = func(
                    *args,
                    batch_size=batch_size,
                    disable_batch_size_cache=disable_batch_size_cache,
                    **kwargs,
                )
                gc_cuda()
                if not disable_batch_size_cache:
                    stacktrace = get_traceback(ignore_first=20, stack_context=10)
                    gpu_info = get_gpu_information()
                    total_available_gpu_memory = gpu_info[2] if gpu_info else gpu_info
                    total_available_ram_memory = get_total_available_ram()
                    # TODO add logging
                    print(
                        f"Total available ram memory - {total_available_ram_memory}, Total available gpu memory - {total_available_gpu_memory}\n"
                    )
                    # Set value for next iteration with the input hash
                    cache.set_value(
                        value=batch_size,
                        stacktrace=stacktrace,
                        total_available_gpu_memory=total_available_gpu_memory,
                        total_available_ram_memory=total_available_ram_memory,
                    )
                return res
            except RuntimeError as exception:
                if batch_size > 1 and should_reduce_batch_size(exception):
                    # add logging
                    print(
                        f"Unable to fit batch size - {batch_size}, Reducing batch size to - {batch_size // 2}"
                    )
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
    """Checks whether batch size can be reduced or not"""
    return (
        is_cuda_out_of_memory(exception)
        or is_cudnn_snafu(exception)
        or is_out_of_cpu_memory(exception)
    )


def is_cuda_out_of_memory(exception):
    """Checks for CUDA OOM Error"""
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA out of memory." in exception.args[0]
    )


def is_cudnn_snafu(exception):
    """For/because of https://github.com/pytorch/pytorch/issues/4107"""
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


def is_out_of_cpu_memory(exception):
    """Checks for CPU OOM Error"""
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


def get_gpu_information() -> Union[None, Tuple[int, float, float]]:
    """Get CUDA gpu related info if gpu exist

    Returns:
        Union[None, Tuple[int, float, float]]: total available gpus, total occupied memory gb, total available gpu memeory
        `None` if unable to get CUDA GPU related info
    """
    try:
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

            return (
                total_available_gpus,
                total_occupied_memory_gb,
                total_available_memory,
            )
        except Exception:
            exc_traceback = traceback.format_exc()
            print(f"Unable to get details of gpu - {exc_traceback}")
            raise
        finally:
            pynvml.nvmlShutdown()
    except BaseException:
        return None


def get_total_available_ram() -> int:
    """Get total available ram in GB

    Returns:
        float: available ram in GB
    """
    memory = psutil.virtual_memory()
    total_available_ram = memory.available

    # Round available RAM to the nearest GB
    total_available_ram_gb = math.ceil(total_available_ram / (1024**3))

    return total_available_ram_gb
