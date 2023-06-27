import math
import random
import warnings
import collections
from operator import itemgetter

import langchain
import numpy as np
import sklearn
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from typing import List, Union, Tuple, Literal, Dict
from llmsearch.utils.model_utils import batcher, infer_data, encoder_decoder_parser, decoder_parser

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Callable

def clone_monkey_patch(estimator, *, safe=True):
    assert not hasattr(estimator, 'model_generation_param_keys'), f"Hyperparameters to tune already defined - {estimator.model_generation_param_keys}"
    return estimator

# monkey patch sklearn clone
sklearn.base.clone = clone_monkey_patch

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
        self.optimal_batch_size = batch_size
        self.disable_batch_size_cache = disable_batch_size_cache
        self.tokenizer_encoding_kwargs = tokenizer_encoding_kwargs
        self.tokenizer_decoding_kwargs = tokenizer_decoding_kwargs
        self.pred_function = pred_function
        # Set generation params
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def fit(self, *args, **kwargs):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if self.pred_function:
            return self.pred_function(X)
        model_generation_params = {
            attr: getattr(self, attr) for attr in self.model_generation_param_keys
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
        Only supported in sklearn >= 1.3

        Returns:
            BaseEstimator: self
        """
        return self

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
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = collections.defaultdict(dict)  # grouped by prefix
        self.model_generation_param_keys = params.keys()
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            # TODO(1.4): remove specific handling of "base_estimator".
            # The "base_estimator" key is special. It was deprecated and
            # renamed to "estimator" for several estimators. This means we
            # need to translate it here and set sub-parameters on "estimator",
            # but only if the user did not explicitly set a value for
            # "base_estimator".
            if (
                key == "base_estimator"
                and valid_params[key] == "deprecated"
                and self.__module__.startswith("sklearn.")
            ):
                warnings.warn(
                    (
                        f"Parameter 'base_estimator' of {self.__class__.__name__} is"
                        " deprecated in favor of 'estimator'. See"
                        f" {self.__class__.__name__}'s docstring for more details."
                    ),
                    FutureWarning,
                    stacklevel=2,
                )
                key = "estimator"

            valid_params[key].set_params(**sub_params)

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
        y_true = dataset_to_evaluate['X']
        y_pred = infer_data(model=self.estimator.model, tokenizer=self.tokenizer,is_encoder_decoder = self.estimator.is_encoder_decoder,batch_size=self.estimator.optimal_batch_size, device=self.device, model_inputs=dataset_to_evaluate['X'], tokenizer_encoding_kwargs=self.tokenizer_encoding_kwargs,tokenizer_decoding_kwargs=self.tokenizer_decoding_kwargs, generation_kwargs=generation_kwargs, disable_batch_size_cache=self.disable_batch_size_cache)
        score = self.score_func(y_true = y_true, y_pred = y_pred)
        return score, y_pred

    def get_tokenizer_quantile(self, input_list, tokenizer_length_percentile = None):
        """Get max length - we take it as a quantile of the input data, default - tokenizer_length_percentile"""
        tokenizer_length_percentile = self.tokenizer_length_percentile if tokenizer_length_percentile is None else tokenizer_length_percentile
        input_ids = self.tokenizer(input_list, max_length = None, truncation = False, padding = False)["input_ids"]
        batch_ids = list(map(len, input_ids))
        return int(np.quantile(batch_ids, q=tokenizer_length_percentile))