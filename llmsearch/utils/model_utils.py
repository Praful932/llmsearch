"""
Common Utilties for Models
"""
import gc
import math
import random
import warnings
from itertools import islice
from typing import List, Dict, Union, Tuple, Iterable, Iterator

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from llmsearch.utils.mem_utils import batch_without_oom_error, gc_cuda

# generation parameters required for sampling
sampling_generation_keys = {"temperature", "top_k", "top_p"}


def strip_output(s: str):
    """Just calls .strip nothing else"""
    return s.strip()


def get_device() -> str:
    """Get device one of "cpu", "cuda", "mps"

    Returns:
        str: device str
    """
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int):
    """Seed for reproducibilty"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@batch_without_oom_error
def infer_data(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    is_encoder_decoder: bool,
    batch_size: int,
    disable_batch_size_cache: bool,  # pylint: disable=unused-argument
    device: str,
    model_inputs: List,
    tokenizer_encoding_kwargs: Dict,
    generation_kwargs: Dict = None,
    tokenizer_decoding_kwargs: Dict = None,
    disable_warnings: bool = False,
    return_optimal_batch_size: bool = False,
    output_prepoc : callable = strip_output,
) -> Union[Tuple[List, int], List]:
    """Infer on data with a specific batch size

    Args:
        model (AutoModelForSeq2SeqLM): model with a `.generate` method
        tokenizer (AutoTokenizer): tokenizer to tokenize the input
        is_encoder_decoder (bool): whether the model is an encoder-decoder model, `False` if not
        batch_size (int): batch_size to run inference with, this gets dynamically reduced if the inference function encounters OOM errors
        disable_batch_size_cache (bool): If `True` for each cross validation run, the pre-defined `batch_size` is used, this could lead to wasted computation time if OOM is raised
        model_inputs (List): model inputs to do inference on
        tokenizer_encoding_kwargs (Dict): Encoding arguments for the `tokenizer`
        generation_kwargs (Dict, optional): generation kwargs to use while generating the output. Defaults to None.
        tokenizer_decoding_kwargs (_type_, optional): Decoding arguments for the `tokenizer`. Defaults to `{'skip_special_tokens' : True}`.
        disable_warnings (bool, optional): disables warnings related to generation parameters. Defaults to False.
        return_optimal_batch_size (bool, optional): if the function should return the optimal batch size found, useful for caching when performing cross validation. Defaults to False.
        output_prepoc (Callable, optional): Prepoc to run on the completion, by default strips the output. Note that this is applied on the completion. Defaults to False.

    Returns:
        Union[Tuple[List, int], List]: outputs and or best batch size
    """
    assert isinstance(
        tokenizer_encoding_kwargs, Dict
    ), f"Incorrect tokenizer kwargs input, expected Dict - {tokenizer_encoding_kwargs}"

    tokenizer_decoding_kwargs = (
        tokenizer_decoding_kwargs
        if tokenizer_decoding_kwargs
        else {"skip_special_tokens": True}
    )

    outputs = []

    # TODO : Make this a separate function
    if any(item in sampling_generation_keys for item in generation_kwargs.keys()):
        sampling_generation_keys_input = sampling_generation_keys.intersection(
            set(generation_kwargs.keys())
        )
        # https://github.com/huggingface/transformers/issues/22405
        if "do_sample" not in generation_kwargs:
            if not disable_warnings:
                warnings.warn(
                    message=f"Invalid generation settings, set `do_sample` parameter to make parameters like {sampling_generation_keys_input} work",
                    stacklevel=3,
                )
        if "generation_seed" not in generation_kwargs:
            if not disable_warnings:
                warnings.warn(
                    message="Generation seed not found in generation parameters, add `generation_seed` in `generation_kwargs` to ensure reproducibility for parameter search.",
                    stacklevel=3,
                )
        elif "do_sample" in generation_kwargs:
            seed_everything(seed=generation_kwargs.pop("generation_seed"))

    for batch in tqdm(
        batcher(iterable=model_inputs, batch_size=batch_size),
        total=math.ceil(len(model_inputs) / batch_size),
    ):
        gc.collect()
        gc_cuda()
        encoded_input = tokenizer(
            text=batch, **tokenizer_encoding_kwargs, return_tensors="pt"
        )
        input_ids = encoded_input.input_ids.to(device)
        attention_mask = encoded_input.attention_mask.to(device)
        output_ids = model.generate(
            inputs=input_ids, attention_mask=attention_mask, **generation_kwargs
        )
        decoded_output = tokenizer.batch_decode(
            sequences=output_ids,
            **tokenizer_decoding_kwargs,
        )
        # remove prompt
        if not is_encoder_decoder:
            decoded_output = decoder_parser(
                outputs=decoded_output, formatted_prompts=batch, prepoc= output_prepoc
            )
        else:
            decoded_output = encoder_decoder_parser(outputs=decoded_output, prepoc=output_prepoc)
        outputs.extend(decoded_output)
    if return_optimal_batch_size:
        return outputs, batch_size
    return outputs


def batcher(iterable: Iterable, batch_size: int) -> Iterator:
    """Batch a iterable into batches of `batch_size`

    Args:
        iterable (Iterable): iterable
        batch_size (int): batch size

    Yields:
        Iterator: iterator over batches
    """
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def encoder_decoder_parser(outputs: str, prepoc : callable):
    """Applies the prepoc function on the completion"""
    return [prepoc(output) for output in outputs]


def decoder_parser(outputs: List[str], formatted_prompts: List[str], prepoc : callable):
    """Removes the promot from the text and calls prepoc on the completion"""
    return [
        prepoc(output[len(formatted_prompt) :])
        for output, formatted_prompt in zip(outputs, formatted_prompts)
    ]
