"""
Main Tuner Module containing EstimatorWrapper Class & Tuner Class for scikit-learn
"""

import inspect
import random
import warnings
import collections
from operator import itemgetter
from typing import List, Union, Tuple, Literal, Dict, Callable

import sklearn
import langchain
import numpy as np
from datasets import Dataset
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmsearch.utils.model_utils import infer_data

import inspect

def print_call_stack(n):
    stack = inspect.stack()[1:n+1]
    for level, frame_info in enumerate(stack[::-1]):
        frame = frame_info.frame
        function_name = frame.f_code.co_name
        filename = frame.f_code.co_filename
        line_number = frame.f_lineno
        print(f"Level {level}: {function_name} | File: {filename}, Line: {line_number}")



def clone_monkey_patch(estimator : BaseEstimator, *, safe : bool=True) -> BaseEstimator:
    """Monkey Patch function to clone the Estimator while doing cross validation/hyperparameter search

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
    print(f"in monkey patch - {estimator}")
    assert not hasattr(estimator, 'model_generation_param_keys'), f"Hyperparameters to tune already defined - {estimator.model_generation_param_keys}"
    # print_call_stack(3)
    return estimator

# monkey patch sklearn clone
# sklearn.base.clone = clone_monkey_patch

class EstimatorWrapper(BaseEstimator):
    def __init__(
        self,
        model,
        tokenizer,
        is_encoder_decoder,
        device,
        scorer,
        batch_size,
        disable_batch_size_cache,
        tokenizer_encoding_kwargs,
        tokenizer_decoding_kwargs = {'skip_special_tokens' : True},
        pred_function = None,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder
        self.device = device
        self.scorer = scorer
        self.batch_size = batch_size
        self._optimal_batch_size = batch_size
        self.disable_batch_size_cache = disable_batch_size_cache
        self.tokenizer_encoding_kwargs = tokenizer_encoding_kwargs
        self.tokenizer_decoding_kwargs = tokenizer_decoding_kwargs
        self.pred_function = pred_function
        # Set generation params, All kwargs are assumed to be generation params
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        self._model_generation_param_keys = list(kwargs.keys())

    def fit(self, *args, **kwargs):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if self.pred_function:
            return self.pred_function(X)
        model_generation_params = {
            attr: getattr(self, attr) for attr in self._model_generation_param_keys
        }
        output, self.optimal_batch_size = infer_data(
            model=self.model,
            tokenizer=self.tokenizer,
            is_encoder_decoder = self.is_encoder_decoder,
            batch_size=self.batch_size,
            disable_batch_size_cache=self.disable_batch_size_cache,
            device=self.device,
            model_inputs=X,
            tokenizer_encoding_kwargs=self.tokenizer_encoding_kwargs,
            tokenizer_decoding_kwargs = self.tokenizer_decoding_kwargs,
            generation_kwargs=model_generation_params,
            return_optimal_batch_size=True,
        )
        return output

    def __sklearn_clone__(self) -> BaseEstimator:
        """Returns the same estimator
        Only relevant from sklearn >= 1.3

        Returns:
            BaseEstimator: self
        """
        print("calling overriden clone")
        return self

    def get_params(self, deep = True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key, value in vars(self).items():
            if not (key.startswith("__") or key.startswith("_")):
                # print(f'in get - {key} - {value}')
                out[key] = value

        # if hasattr(self, 'model_generation_param_keys'):
        #     for key in self.model_generation_param_keys:
        #         value = getattr(self, key)
        #         out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        # print("in set params")
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        self._model_generation_param_keys = list(params.keys())

        for key, value in params.items():
            setattr(self, key, value)

        return self


class Tuner:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt_template : langchain.BasePromptTemplate,
        dataset: Union[Dataset, Dict],
        device: str,
        scorer : Callable,
        is_encoder_decoder,
        greater_is_better : bool = True,
        seed: int = 42,
        tokenizer_encoding_kwargs: Dict = None,
        tokenizer_decoding_kwargs : Dict = {'skip_special_tokens' : True},
        batch_size: int = 32,
        disable_batch_size_cache : bool = False,
        sample_ratio: float = 0.3,
        tokenizer_length_percentile: float = 0.9,
    ):
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.dataset = dataset.map(lambda sample : {'X' : self.prompt_template.format(X=sample['X']), 'y' : sample['y']})
        self.device = device
        self.seed = seed
        self.score_func = scorer
        self.scorer = make_scorer(score_func = scorer, greater_is_better = greater_is_better)
        self.disable_batch_size_cache = disable_batch_size_cache
        self.sample_ratio = sample_ratio
        self.tokenizer_length_percentile = tokenizer_length_percentile
        self.tokenizer_encoding_kwargs = self.get_default_input_tokenizer_kwargs(
            sample_ratio=sample_ratio,
            tokenizer_kwargs=tokenizer_encoding_kwargs,
        )
        self.tokenizer_decoding_kwargs = tokenizer_decoding_kwargs
        self.estimator = EstimatorWrapper(
            model=model,
            tokenizer=self.tokenizer,
            is_encoder_decoder=is_encoder_decoder,
            device=self.device,
            scorer = self.scorer,
            batch_size=batch_size,
            disable_batch_size_cache=disable_batch_size_cache,
            tokenizer_encoding_kwargs=self.tokenizer_encoding_kwargs,
            tokenizer_decoding_kwargs=self.tokenizer_decoding_kwargs,
        )

    def get_default_input_tokenizer_kwargs(
        self,
        sample_ratio: float,
        tokenizer_kwargs: Union[Dict, None],
    ):
        """Get default input tokenizer kwargs

        Args:
            sample_ratio (float): _description_
            tokenizer_length_percentile (float): _description_
            tokenizer_kwargs (Union[Dict, None]): _description_
        """

        if tokenizer_kwargs:
            return tokenizer_kwargs

        tokenizer_encoding_kwargs = {
            "padding": True,
            "truncation": True,
        }
        sample_size = int(len(self.dataset["y"]) * sample_ratio)
        random.seed(self.seed)
        sample_indexes = random.sample(range(0, len(self.dataset["y"])), sample_size)
        get_items = itemgetter(*sample_indexes)
        X = get_items(self.dataset["X"])
        tokenizer_encoding_kwargs["max_length"] = self.get_tokenizer_quantile(input_list = X, tokenizer_length_percentile = self.tokenizer_length_percentile)
        return tokenizer_encoding_kwargs

    def get_score(self, generation_kwargs, dataset = None) -> Tuple[float, List]:
        dataset_to_evaluate = dataset if dataset else self.dataset
        y_true = dataset_to_evaluate['y']
        y_pred = infer_data(model=self.estimator.model, tokenizer=self.tokenizer,is_encoder_decoder = self.estimator.is_encoder_decoder,batch_size=self.estimator.optimal_batch_size, device=self.device, model_inputs=dataset_to_evaluate['X'], tokenizer_encoding_kwargs=self.tokenizer_encoding_kwargs,tokenizer_decoding_kwargs=self.tokenizer_decoding_kwargs, generation_kwargs=generation_kwargs, disable_batch_size_cache=self.disable_batch_size_cache)
        score = self.score_func(y_true = y_true, y_pred = y_pred)
        return score, y_pred

    def get_tokenizer_quantile(self, input_list, tokenizer_length_percentile = None):
        """Get max length - we take it as a quantile of the input data, default - tokenizer_length_percentile"""
        tokenizer_length_percentile = self.tokenizer_length_percentile if tokenizer_length_percentile is None else tokenizer_length_percentile
        input_ids = self.tokenizer(input_list, max_length = None, truncation = False, padding = False)["input_ids"]
        batch_ids = list(map(len, input_ids))
        return int(np.quantile(batch_ids, q=tokenizer_length_percentile))