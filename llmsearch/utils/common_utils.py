"""
Common utils
"""

import inspect
from sklearn.base import BaseEstimator


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
    estimator: BaseEstimator, *, safe: bool = True
) -> BaseEstimator:  # pylint: disable=unused-argument
    """Deprecated Monkey Patch function to clone the Estimator while doing cross validation/hyperparameter search

    Usable in <= 1.3 versions of scikit-learn versions

    - This functions returns the same estimator, as there are no parameters to specifically "fit"
    - This is done to avoid OOM errors for larger models, this does not affect the hyperparameter search in any way
    - The monkey patch will be deprecated when this package supports scikit-learn - >= 1.3.0 (atm not released)
        - https://scikit-learn.org/dev/whats_new/v1.3.html#changelog:~:text=A%20__sklearn_clone__%20protocol%20is%20now%20available%20to%20override%20the%20default%20behavior%20of%20base.clone.

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
