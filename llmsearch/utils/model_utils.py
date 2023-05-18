"""
Common Utilties for Models
"""
import random
import math
import gc
import time
from itertools import islice
from typing import List, Dict, Union, Iterable, Iterator

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from llmsearch.utils.mem_utils import batch


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
TODO
1. decorator that is being used is not correct, understand how parametrized decorator works
2. write a custom decorator which takes that into account
"""

@batch
def infer_data(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size : int,
    model_inputs: List,
    model_input_tokenizer_kwargs: Dict,
    generation_kwargs: Dict,
) -> Union[List, float]:
    """Infer on data with a specific batch size

    Args:
        model (AutoModelForSeq2SeqLM): model with a generate attribute
        tokenizer (AutoTokenizer): tokenizer to tokenize the input
        device (str): device to run the model on
        model_inputs (List): model inputs to do inference on
        model_input_tokenizer_kwargs (Dict): tokenizer params to tokenize the input
        generation_kwargs (Dict): generation kwargs to use while generating the output
        batch_size (int): batch size for `model_inputs`

    Returns:
        Union[List, float]: _description_
    """
    assert isinstance(model_input_tokenizer_kwargs, Dict), f"Incorrect tokenizer kwargs input, expected Dict - {model_input_tokenizer_kwargs}"
    start = time.time()
    outputs = []
    for batch in tqdm(
        batcher(iterable=model_inputs, batch_size=batch_size),
        total=math.ceil(len(model_inputs) / batch_size),
    ):
        gc.collect()
        input_ids = tokenizer(
            text=batch, **model_input_tokenizer_kwargs, return_tensors="pt"
        ).input_ids.to(device)
        output_ids = model.generate(input_ids, **generation_kwargs)
        decoded_output = tokenizer.batch_decode(
            sequences=output_ids, skip_special_tokens=True
        )
        outputs.extend(decoded_output)
    end = time.time()
    total_latency = (end - start) * 1000
    return outputs, total_latency


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
