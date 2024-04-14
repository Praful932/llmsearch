"""
Common utilties
"""

import json
import yaml
import inspect
from pathlib import Path
from sklearn.base import BaseEstimator

def json_dump(ob : dict, file_path: Path):
    with open(file_path, 'w', encoding="utf-8") as json_file:
        json.dump(ob, json_file, indent=4)

def json_load(file_path: Path):
    with open(file_path, 'r', encoding="utf-8") as json_file:
        return json.load(json_file)

def yaml_load(file_path):
    """Load yaml file from file path

    Args:
        file_path (str): path to yaml file

    Returns:
        dict: loaded yaml file
    """
    with open(file_path, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def print_call_stack(n: int):
    """Prints call stack from first call to latest call

    Args:
        n (int): last n calls, last call would be the latest one
    """
    stack = inspect.stack()[1 : n + 1]
    for level, frame_info in enumerate(stack[::-1]):
        frame = frame_info.frame
        function_name = frame.f_code.co_name
        filename = frame.f_code.co_filename
        line_number = frame.f_lineno
        print(f"Level {level}: {function_name} | File: {filename}, Line: {line_number}")


def clone_monkey_patch(
    estimator: BaseEstimator, *, safe: bool = True  # pylint: disable=unused-argument
) -> BaseEstimator:
    """Deprecated Monkey Patch function to clone the Estimator while doing cross validation/hyperparameter search

    Usable in < 1.3 versions of scikit-learn versions

    - This functions returns the same estimator, as there are no parameters to specifically "fit"
    - This is done to avoid OOM errors for larger models, this does not affect the hyperparameter search in any way

    Args:
        estimator (BaseEstimator): estimator to clone
        safe (bool, optional): redundant parameter for now. Defaults to True.

    Returns:
        BaseEstimator: returns the same estimator
    """
    assert not hasattr(
        estimator, "model_generation_param_keys"
    ), f"Hyperparameters to tune already defined - {estimator.model_generation_param_keys}"
    return estimator
