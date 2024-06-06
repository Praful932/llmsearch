"""
Generation Related Utilties\n
**Note** : Generation Type Detection Scripts are updated as of transformers `v4.31.0`
"""

from typing import Dict, Union, List, Tuple

from llmsearch.utils.logging_utils import get_logger

logger = get_logger(__name__)

ops = ["equal", "lt", "gt", "ne"]

# Default generation parameters
gen_param_defaults = {
    "do_sample": False,
    "num_beams": 1,
    "num_beam_groups": 1,
    "penalty_alpha": None,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "typical_p": 1.0,
    "epsilon_cutoff": 0.0,
    "eta_cutoff": 0.0,
    "diversity_penalty": 0.0,
    "repetition_penalty": 1.0,
    "encoder_repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "bad_words_ids": None,
    "force_words_ids": None,
    "renormalize_logits": False,
    "constraints": None,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "remove_invalid_values": None,
    "exponential_decay_length_penalty": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "forced_decoder_ids": None,
    "sequence_bias": None,
    "guidance_scale": None,
    "encoder_no_repeat_ngram_size": 0,
}


# Generation type mapping
gen_type_map = {
    "Contrastive Search Decoding": {
        "num_beams": {"equal": 1},
        "top_k": {"ne": None, "gt": 1},
        "do_sample": {"equal": False},
        "penalty_alpha": {"ne": None, "gt": 0},
    },
    "Greedy Decoding": {
        "num_beams": {"equal": 1},
        "num_beam_groups": {"equal": 1},
        "do_sample": {"equal": False},
    },
    "Sampling": {
        "num_beams": {"equal": 1},
        "num_beam_groups": {"equal": 1},
        "do_sample": {"equal": True},
    },
    "Beam Search": {
        "num_beams": {"gt": 1},
        "num_beam_groups": {"equal": 1},
        "do_sample": {"equal": False},
    },
    "Beam Search with sampling": {
        "num_beams": {"gt": 1},
        "num_beam_groups": {"equal": 1},
        "do_sample": {"equal": True},
    },
    "Group Beam Search": {
        "num_beams": {"gt": 1},
        "num_beam_groups": {"gt": 1},
        "diversity_penalty": {"gt": 0},
        "do_sample": {"equal": False},
    },
    "Mirostat Sampling": {
        "mirostat_mode": {"equal": 2},
        "do_sample": {"equal": True},
    },
    "Tail-Free Sampling": {
        "tfs": {"gte": 0, "lte": 1.0},
        "do_sample": {"equal": True},
    },
    "Top-A Sampling": {
        "top_a": {"gte": 0, "lte": 1.0},
        "do_sample": {"equal": True},
    },
}

# Generation parameters dependent on 'do_sample` Parameter
param_dependent_on_sampling = [
    "temperature",
    "top_k",
    "top_p",
    "typical_p",
    "eta_cutoff",
    "mirostat_mode",
    "tfs",
    "top_a",
]


def check_sample_parameter(gen_params: Dict, gen_type_params: Union[None, List]):
    """
    Check if the `do_sample` parameter is set to True when other parameters that are dependent on sampling are present.

    Args:
        gen_params (Dict): Dictionary containing generation parameters.
        gen_type_params (Dict): Params to exclude if already being used by a generation type (`top_k` in Contrastive Search Decoding)

    Raises:
        UserWarning: Warns if dependent generation parameters are present without 'do_sample' set to True.
    """
    if not gen_params.get("do_sample", gen_param_defaults["do_sample"]) and any(
        param in gen_params.keys() for param in param_dependent_on_sampling
    ):
        params = [
            param
            for param in gen_params.keys()
            if param in param_dependent_on_sampling
            if param not in gen_type_params
        ]
        if params:
            logger.warning(
                "Sampling dependent generation parameter/s present - %s, please set `do_sample` to True, for these params to work.",
                params,
            )


def identify_and_validate_gen_params(gen_params: Dict):
    """
    Identify and validate the generation type based on provided generation parameters.

    Args:
        gen_params (Dict): Dictionary containing generation parameters.

    Returns:
        str: Name of the identified generation type.
    """
    ret_val = "unable_to_detect"
    for gen_type_name, rules in gen_type_map.items():
        check = check_if_gen_param_rules_satisfy(gen_params=gen_params, rules=rules)
        if check:
            check_sample_parameter(
                gen_params=gen_params,
                gen_type_params=rules.keys(),
            )
            ret_val = gen_type_name
            break
    return ret_val


def check_if_gen_param_rules_satisfy(gen_params: Dict, rules: Dict):
    """
    Check if the provided generation parameters satisfy the specified rules.

    Args:
        gen_params (Dict): Dictionary containing generation parameters.
        rules (Dict): Dictionary containing rules for parameter validation.

    Returns:
        bool: True if all rules are satisfied, False otherwise.
    """
    checks = []
    for param, param_req in rules.items():
        check = check_if_param_req_satisfy(
            gen_params=gen_params, param=param, param_req=param_req
        )
        checks.append(check)
    return sum(checks) == len(checks)


def check_if_param_req_satisfy(gen_params: Dict, param: str, param_req: Dict):
    """
    Check if a specific generation parameter satisfies the specified requirement.

    Args:
        gen_params (Dict): Dictionary containing generation parameters.
        param (str): Name of the generation parameter to check.
        param_req (Dict): Dictionary specifying the requirement for the parameter.

    Returns:
        bool: True if the requirement is satisfied, False otherwise.
    """
    checks = []
    for op, value_to_check in param_req.items():
        check = False
        assert op in ops, f"Invalid operation - {op}, not in defined ops - {ops}"
        value = gen_params.get(param, gen_param_defaults[param])
        if op == "equal":
            check = value == value_to_check
        elif op == "ne":
            check = value != value_to_check
        elif op == "gt" and value is not None:
            check = value > value_to_check
        elif op == "lt" and value is not None:
            check = value < value_to_check
        checks.append(check)
    return bool(sum(checks))


def get_sample_hyp_space(seed: int, max_new_tokens: int) -> Tuple[List, List]:
    """Get 2 sample hyp spaces

    Args:
        seed (int): seed

    Returns:
        Tuple[List, List]: First Item is a larger hyp space which searches for individual generation types, Second Item of the Tuple are the top generation params as evaluated by [oobabooga using Vicuna-13B with instruct prompts](https://www.reddit.com/r/LocalLLaMA/comments/14adfw2/preset_arena_17205_comparisons_between_241/).
    """
    param_grid1 = [
        {
            "num_beams": [1],
            "top_k": list(range(4, 20)),
            "do_sample": [True],
            "penalty_alpha": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
        {
            "num_beams": [1],
            "num_beam_groups": [1],
            "do_sample": [False],
            "max_new_tokens": [max_new_tokens],
        },
        {
            "num_beams": [3],
            "num_beam_groups": [1],
            "do_sample": [False],
            "max_new_tokens": [max_new_tokens],
        },
        {
            "num_beams": list(range(2, 4)),
            "num_beam_groups": [1],
            "do_sample": [False],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
            "no_repeat_ngram_size": [0, 3, 4, 5],
        },
        # disabling as currently not supported
        # {
        #     "mirostat_mode": [2],
        #     "mirostat_eta": [0.1, 0.2, 0.3, 0.4, 0.5],
        #     "mirostat_tau": list(range(3, 8)),
        #     "do_sample": [True],
        #     "generation_seed": [seed],
        #     "max_new_tokens" : [max_new_tokens],
        # },
        {
            "tfs": [0.8, 0.85, 0.9, 0.95, 0.99],
            "do_sample": [True],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
        {
            "top_a": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "do_sample": [True],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
        {
            # exhaustive search
            "top_p": [0.5, 0.7, 0.9, 1.0],
            "temperature": [0.7, 0.8, 0.9, 1.0],
            "no_repeat_ngram_size": [0, 3, 4, 5],
            "repetition_penalty": [1.0, 1.1, 1.2],
            "do_sample": [True],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
    ]
    # presets as found in https://github.com/oobabooga/oobabooga.github.io/blob/main/arena/results.md#presets-that-i-chose
    param_grid_2 = [
        {
            # random_preset_066
            "epsilon_cutoff": [1.49],
            "eta_cutoff": [10.42],
            "repetition_penalty": [1.17],
            "temperature": [1.31],
            "top_a": [0.52],
            "top_k": [49],
            "top_p": [0.14],
            "do_sample": [True],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
        {
            # random_preset_134
            "repetition_penalty": [1.01],
            "temperature": [0.87],
            "tfs": [0.68],
            "top_k": [85],
            "top_p": [0.99],
            "typical_p": [0.68],
            "do_sample": [True],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
        {
            # simple-1
            "repetition_penalty": [1.15],
            "temperature": [0.7],
            "top_k": [20],
            "top_p": [0.9],
            "do_sample": [True],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
        {
            # random_preset_035
            "repetition_penalty": [1.09],
            "temperature": [1.31],
            "top_k": [72],
            "top_p": [0.29],
            "do_sample": [True],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
        {
            # starchat
            "temperature": [0.2],
            "top_k": [50],
            "top_p": [0.95],
            "do_sample": [True],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
        {
            # random_preset_183
            "encoder_repetition_penalty": [1.07],
            "eta_cutoff": [10.78],
            "repetition_penalty": [1.21],
            "temperature": [1.01],
            "top_a": [0.75],
            "top_k": [91],
            "top_p": [0.21],
            "do_sample": [True],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
        {
            # tfs-with-top-a
            "repetition_penalty": [1.15],
            "temperature": [0.7],
            "tfs": [0.95],
            "top_a": [0.2],
            "do_sample": [True],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
        {
            # Special-Contrastive Search-3
            "do_sample": [False],
            "penalty_alpha": [0.3],
            "top_k": [4],
            "max_new_tokens": [max_new_tokens],
        },
        {
            "repetition_penalty": [1.02],
            "temperature": [1.68],
            "tfs": [0.97],
            "top_a": [0.42],
            "top_k": [77],
            "top_p": [0.17],
            "do_sample": [True],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
        # disabling as currently not supported
        # {
        #     # mirostat
        #     "mirostat_mode": [2],
        #     "mirostat_eta": [0.1],
        #     "mirostat_tau": [5],
        #     "do_sample": [True],
        #     "generation_seed": [seed],
        #     "max_new_tokens" : [max_new_tokens],
        # },
        {
            # NovelAI-Ouroboros
            "repetition_penalty": [1.05],
            "temperature": [1.07],
            "top_k": [100],
            "do_sample": [True],
            "generation_seed": [seed],
            "max_new_tokens": [max_new_tokens],
        },
    ]
    return param_grid1, param_grid_2
