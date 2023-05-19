import math
import random
import warnings
import collections
from operator import itemgetter

import numpy as np

from tqdm.auto import tqdm
from sklearn.base import BaseEstimator
from typing import List, Union, Tuple, Literal, Dict
from llmsearch.utils.model_utils import batcher, infer_data

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

class EstimatorWrapper(BaseEstimator):
    def __init__(self, model, tokenizer, device, batch_size, model_input_tokenizer_kwargs, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.model_input_tokenizer_kwargs = model_input_tokenizer_kwargs
        # Set generation params
        for k,v in kwargs.items():
            self.__setattr__(k, v)

    def fit(self, *args, **kwargs):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        model_generation_params = {
            attr : getattr(self, attr) for attr in self.model_generation_param_keys
        }
        output, _ = infer_data(
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            device = self.device,
            model_inputs=X,
            model_input_tokenizer_kwargs=self.model_input_tokenizer_kwargs,
            generation_kwargs=model_generation_params,
        )
        return output


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
    def __init__(self, model : AutoModelForCausalLM, tokenizer : AutoTokenizer, dataset : Union[Dataset, Dict], device : str, seed : int = 42, model_input_tokenizer_kwargs : Dict = None, batch_size : int = 32, sample_ratio : float = 0.3, tokenizer_length_percentile : float = 0.9):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        self.seed = seed
        self.model_input_tokenizer_kwargs = self.get_default_input_tokenizer_kwargs(sample_ratio = sample_ratio,tokenizer_length_percentile=tokenizer_length_percentile, tokenizer_kwargs=model_input_tokenizer_kwargs)
        self.sample_ratio = sample_ratio
        self.tokenizer_length_percentile = tokenizer_length_percentile
        self.estimator = EstimatorWrapper(model=model, tokenizer = tokenizer, device=device, batch_size=batch_size, model_input_tokenizer_kwargs=model_input_tokenizer_kwargs)


    def get_default_input_tokenizer_kwargs(self, sample_ratio : float,tokenizer_length_percentile : float, tokenizer_kwargs : Union[Dict, None]):
        """Get default input tokenizer kwargs

        Args:
            sample_ratio (float): _description_
            tokenizer_length_percentile (float): _description_
            tokenizer_kwargs (Union[Dict, None]): _description_
        """
        def get_max_length(X : Dict) -> float:
            """Get max length - we take it as a quantile of the input data, default - tokenizer_length_percentile
            """
            batch_input_ids = self.tokenizer(X, max_length=None, truncation=False, padding=False)['input_ids']
            batch_input_ids = list(map(len, batch_input_ids))
            return np.quantile(batch_input_ids, q = tokenizer_length_percentile)

        if tokenizer_kwargs:
            return tokenizer_kwargs

        model_input_tokenizer_kwargs = {
            'padding' : True,
            'truncation' : True,
        }
        sample_size = len(self.dataset['y']) * sample_ratio
        random.seed(self.seed)
        sample_indexes = random.randint(range(0, len(self.dataset['y'])), sample_size)
        get_items = itemgetter(*sample_indexes)
        X = get_items(self.dataset['X'])
        model_input_tokenizer_kwargs['max_length'] = get_max_length(X)
        return model_input_tokenizer_kwargs










