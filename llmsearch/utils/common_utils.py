"""
Common utilties
"""

import json
import yaml
import inspect
from pathlib import Path
from sklearn.base import BaseEstimator
from datasets import Dataset


def json_dump(ob: dict, file_path: Path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(ob, json_file, indent=4)


def json_load(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as json_file:
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


def process_dataset_with_map(
    dataset, sample_preprocessor, tokenizer, input_cols, eval_cols
):
    """Processes the given dataset by mapping a processing function over each sample.

    Args:
        dataset (Dataset): A Hugging Face dataset to be processed.
        sample_preprocessor (function): A function to preprocess sample inputs.
        tokenizer (function): A tokenizer function to apply to input text.
        input_cols (list of str): Column names to be processed for input features.
        eval_cols (list of str): Column names for evaluation labels.

    Returns:
        Dataset: A new dataset with original data and additional processed keys `_X` and `_y`.
    """

    def process_sample(sample):
        # Copy original data to retain all keys
        sample_dict = dict(sample)
        processed_sample = sample_dict.copy()
        # Process inputs and store in '_X'
        processed_sample["_X"] = sample_preprocessor(
            tokenizer, **{col: sample[col] for col in input_cols + eval_cols}
        )
        # Extract evaluation columns and store in '_y'
        processed_sample["_y"] = {eval_col: sample[eval_col] for eval_col in eval_cols}
        return processed_sample

    # Apply the processing function to each sample in the dataset
    mapped_data = map(process_sample, dataset)
    # Convert the mapped data to a list (necessary for further processing)
    mapped_list = list(mapped_data)

    # Construct a new dataset from the list of processed samples
    # This uses dictionary comprehension to handle potentially complex nested structures in '_X' and '_y'
    return Dataset.from_dict(
        {key: [dic[key] for dic in mapped_list] for key in mapped_list[0]}
    )


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
