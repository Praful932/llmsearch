"""
Common Utilties for Models
"""
import gc
import time
from itertools import islice
from typing import List, Dict, Union, Iterable, Iterator

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def infer_data(
    model: AutoModelForSeq2SeqLM(),
    tokenizer: AutoTokenizer(),
    device: str,
    model_inputs: List,
    model_input_tokenizer_kwargs: Dict,
    generation_kwargs: Dict,
    batch_size: int,
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
    start = time.time()
    outputs = []
    for batch in batcher(iterable=model_inputs, batch_size=batch_size):
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
