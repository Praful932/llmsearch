"""
Generation Related Utilties
"""
from typing import Dict, Union, List

from llmsearch.utils.logging_utils import get_logger

logger = get_logger(__name__)

ops = ["equal", "lt", "gt", "ne"]

logits_process_params = [
    "temperature",
    "top_k",
    "top_p",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",
    "diversity_penalty",
    "repetition_penalty",
    "encoder_repetition_penalty",
    "length_penalty",
    "no_repeat_ngram_size",
    "bad_words_ids",
    "force_words_ids",
    "renormalize_logits",
    "constraints",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "remove_invalid_values",
    "exponential_decay_length_penalty",
    "suppress_tokens",
    "begin_suppress_tokens",
    "forced_decoder_ids",
    "sequence_bias",
    "guidance_scale",
]

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
}

param_dependent_on_sampling = [
    "temperature",
    "top_k",
    "top_p",
    "typical_p",
    "eta_cutoff",
]


def check_sample_parameter(gen_params: Dict, gen_type_params: Union[None, List]):
    """
    Check if the 'do_sample' parameter is set to True when other parameters that are dependent on sampling are present.

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
                gen_type_params=gen_type_map[gen_type_name].keys(),
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
