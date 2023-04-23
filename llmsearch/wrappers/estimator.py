"""
Base Class wrappers
"""

import warnings
from collections import defaultdict

from tqdm.auto import tqdm
from sklearn.base import BaseEstimator

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from llmsearch.utils.model_utils import batcher, infer_data


class EstimatorWrapper(BaseEstimator):
    """Estimator Wrapper for Transformer Models"""

    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer(),
        batch_size: int,
        device: str,
        model_input_tokenizer_kwargs: dict,
        **kwargs,
    ):
        """Initialize Estimator

        Args:
            model (AutoModelForSeq2SeqLM): model to wrap
            tokenizer (AutoTokenizer): tokenizer to wrap
            batch_size (int): _description_
            device (str): _description_
            model_input_tokenizer_kwargs (dict): _description_
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.model_input_tokenizer_kwargs = model_input_tokenizer_kwargs
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    # pylint: disable=W0613
    def fit(self, **args):
        """_summary_

        Args:

        Returns:
            _type_: _description_
        """
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        output = []
        model_generation_params = {
            attr: getattr(self, attr) for attr in self.model_generation_param_keys
        }
        for batch in tqdm(batcher(X, batch_size=self.batch_size)):
            batch_output, _ = infer_data(
                model=self.model,
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                device=self.device,
                model_inputs=batch,
                model_input_tokenizer_kwargs=self.model_input_tokenizer_kwargs,
                generation_kwargs=model_generation_params,
            )
            output.extend(batch_output)
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

        nested_params = defaultdict(dict)  # grouped by prefix
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
