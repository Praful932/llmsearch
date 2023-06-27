"""
Common Utilties for Models
"""
import gc
import math
import time
import random
import warnings
from itertools import islice
from typing import List, Dict, Union, Iterable, Iterator

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from llmsearch.utils.mem_utils import batch

# generation parameters required for sampling
sampling_generation_keys = {'temperature', 'top_k', 'top_p'}

def output_preproc(s : str):
    return s.strip()

def get_device():
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def seed_everything(seed):
    """Seed for reproducibilty"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

"""
do_sample - False
these parameters should be the default value
1. temperature
2. top_k
3. top_p


by default do_sample is False, and it does greedy decoding
"""

@batch
def infer_data(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    is_encoder_decoder : bool,
    batch_size : int,
    disable_batch_size_cache : bool,
    device: str,
    model_inputs: List,
    generation_kwargs: Dict,
    tokenizer_encoding_kwargs: Dict,
    tokenizer_decoding_kwargs : Dict = {'skip_special_tokens' : True},
    disable_warnings : bool = False,
    return_optimal_batch_size : bool = False,
) -> Union[List, float]:
    """Infer on data with a specific batch size

    Args:
        model (AutoModelForSeq2SeqLM): model with a generate attribute
        tokenizer (AutoTokenizer): tokenizer to tokenize the input
        device (str): device to run the model on
        model_inputs (List): model inputs to do inference on
        tokenizer_encoding_kwargs (Dict): tokenizer params to tokenize the input
        generation_kwargs (Dict): generation kwargs to use while generating the output
        batch_size (int): batch size for `model_inputs`

    Returns:
        Union[List, float]: _description_
    """
    assert isinstance(tokenizer_encoding_kwargs, Dict), f"Incorrect tokenizer kwargs input, expected Dict - {tokenizer_encoding_kwargs}"
    outputs = []
    if any(item in sampling_generation_keys for item in generation_kwargs.keys()):
        sampling_generation_keys_input = sampling_generation_keys.intersection(set(generation_kwargs.keys()))
        # https://github.com/huggingface/transformers/issues/22405
        if 'do_sample' not in generation_kwargs:
            if not disable_warnings:
                warnings.warn(message = f"Invalid generation settings, set `do_sample` parameter to make parameters like {sampling_generation_keys_input} work", stacklevel=3)
        if 'generation_seed' not in generation_kwargs:
            if not disable_warnings:
                warnings.warn(message = "Generation seed not found in generation parameters, add `generation_seed` in `generation_kwargs` to ensure reproducibility for parameter search.", stacklevel=3)
        elif 'do_sample' in generation_kwargs:
            seed_everything(seed = generation_kwargs.pop('generation_seed'))
    for batch in tqdm(
        batcher(iterable=model_inputs, batch_size=batch_size),
        total=math.ceil(len(model_inputs) / batch_size),
    ):
        gc.collect()
        encoded_input = tokenizer(
            text=batch, **tokenizer_encoding_kwargs, return_tensors="pt"
        )
        input_ids =encoded_input.input_ids.to(device)
        attention_mask =encoded_input.attention_mask.to(device)
        output_ids = model.generate(inputs=input_ids,attention_mask=attention_mask, **generation_kwargs)
        decoded_output = tokenizer.batch_decode(
            sequences=output_ids, **tokenizer_decoding_kwargs,
        )
        if not is_encoder_decoder:
            decoded_output = decoder_parser(outputs = decoded_output, formatted_prompts=batch)
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

def encoder_decoder_parser(outputs : str):
    return [output_preproc(output) for output in outputs]

def decoder_parser(outputs : List[str],formatted_prompts : List[str]):
    return [output_preproc(output[len(formatted_prompt):]) for output, formatted_prompt in zip(outputs, formatted_prompts)]