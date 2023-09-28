"""
Common Utilties for Models
"""
import gc
import math
import random
from itertools import islice
from typing import List, Dict, Union, Tuple, Iterable, Iterator

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from llmsearch.utils.gen_utils import identify_and_validate_gen_params
from llmsearch.utils.logging_utils import get_logger
from llmsearch.utils.mem_utils import batch_without_oom_error, gc_cuda

logger = get_logger(__name__)


def get_device() -> str:
    """Get device one of "cpu", "cuda", "mps"

    Returns:
        str: device str
    """
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
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
    disable_generation_param_checks: bool = False,
    tokenizer_decoding_kwargs: Dict = None,
    return_optimal_batch_size: bool = False,
    output_preproc: callable = lambda x: x.strip(),
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
        disable_generation_param_checks (bool, optional): Disables the custom generation parameter checks, this check does a sanity check of the parameters & produces warnings before doing generation, Not stable right now.
        tokenizer_decoding_kwargs (_type_, optional): Decoding arguments for the `tokenizer`. Defaults to `{'skip_special_tokens' : True}`.
        return_optimal_batch_size (bool, optional): if the function should return the optimal batch size found, useful for caching when performing cross validation. Defaults to False.
        output_preproc (Callable, optional): Prepoc to run on the completion, by default strips the output. Note that this is applied on the completion. Defaults to False.

    Returns:
        Union[Tuple[List, int], List]: outputs and or best batch size
    """
    seed = None
    assert isinstance(
        tokenizer_encoding_kwargs, Dict
    ), f"Incorrect tokenizer kwargs input, expected Dict - {tokenizer_encoding_kwargs}"

    tokenizer_decoding_kwargs = (
        tokenizer_decoding_kwargs
        if tokenizer_decoding_kwargs
        else {"skip_special_tokens": True}
    )

    outputs = []

    if not disable_generation_param_checks:
        generation_type = identify_and_validate_gen_params(gen_params=generation_kwargs)
        logger.info("Detected generation type - %s", generation_type)

    if "generation_seed" in generation_kwargs:
        seed = generation_kwargs["generation_seed"]
        seed_everything(seed=seed)

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
                outputs=decoded_output, formatted_prompts=batch, prepoc=output_preproc
            )
        else:
            decoded_output = encoder_decoder_parser(
                outputs=decoded_output, prepoc=output_preproc
            )
        outputs.extend(decoded_output)
    for x, y_pred in zip(model_inputs, outputs):
        logger.debug("Input - %s", repr(x))
        logger.debug("Model Output - %s", repr(y_pred))
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


def encoder_decoder_parser(outputs: str, prepoc: callable):
    """Applies the prepoc function on the completion"""
    return [prepoc(output) for output in outputs]


def decoder_parser(outputs: List[str], formatted_prompts: List[str], prepoc: callable):
    """Removes the prompt from the text and calls prepoc on the completion"""
    return [
        prepoc(output[len(formatted_prompt) :])
        for output, formatted_prompt in zip(outputs, formatted_prompts)
    ]
