"""
Main Tuner Module containing LLMEstimatorWrapper Class & Tuner Class for scikit-learn
"""

import random
from operator import itemgetter
from typing import List, Union, Tuple, Dict, Callable

import langchain
import numpy as np
from torch import nn
from datasets import Dataset
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from llmsearch.utils.model_utils import infer_data

from llmsearch.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMEstimatorWrapper(BaseEstimator):
    """Estimator Wrapper Class that abstracts a Model for compatibility with scikit-learn"""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
        device: str,
        scorer: Callable,
        batch_size: int,
        disable_batch_size_cache: bool,
        tokenizer_encoding_kwargs: Dict,
        tokenizer_decoding_kwargs: Dict = None,
        disable_generation_param_checks: bool = False,
        pred_function: Union[Callable, None] = None,
        **kwargs,
    ):
        """Initializes the Estimator

        Args:
            model (nn.Module): model that has a `.generate` method
            tokenizer (AutoTokenizer): tokenizer for the input
            is_encoder_decoder (bool): whether the model is an encoder-decoder model, `False` if not
            device (str): device to run inference on, eg - `cuda:0`
            scorer (Callable): A function that has this signature - `(y_true: List, y_pred: List) -> float` , takes in ground truth and predictions are returns a metric to optimize on
            batch_size (int): batch_size to run inference with, this gets dynamically halfed if the inference function encounters OOM errors
            disable_batch_size_cache (bool): If `True` for each cross validation run, the pre-defined `batch_size` is used, this could lead to wasted computation time if OOM is raised by the inference function
            tokenizer_encoding_kwargs (Dict): Encoding arguments for the `tokenizer`
            tokenizer_decoding_kwargs (Dict, optional): Decoding arguments for the `tokenizer`. Defaults to `{'skip_special_tokens' : True}`
            disable_generation_param_checks (bool, optional): Disables the custom generation parameter checks, this does a sanity check of the parameters & produces warnings before doing generation, Not stable right now.
            pred_function (Union[Callable, None], optional): Override Prediction Function `.predict` is called. The overriden function should have the signature - `(estimator : LLMEstimatorWrapper, model_inputs : List, generation_params : Dict) -> outputs : List` & should return a list of outputs which can be directly consumed by `scorer` as `y_pred`. Defaults to None.

        - All `kwargs` are assumed to be generation params and used when doing Hyperparameter search
        """
        self.model = model
        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder
        self.device = device
        self.scorer = scorer
        self.batch_size = batch_size
        # stores the optimal batch size for a particular configuration if disable_batch_size_cache is `False`
        self._optimal_batch_size = batch_size
        self.disable_batch_size_cache = disable_batch_size_cache
        self.tokenizer_encoding_kwargs = tokenizer_encoding_kwargs
        self.tokenizer_decoding_kwargs = (
            tokenizer_decoding_kwargs
            if tokenizer_decoding_kwargs
            else {"skip_special_tokens": True}
        )
        self.pred_function = pred_function
        self.disable_generation_param_checks = disable_generation_param_checks
        # Set generation params, All kwargs are assumed to be generation params
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        # Stores the names of the keys passed in
        self._model_generation_param_keys = list(kwargs.keys())
        logger.debug(
            "Initializing new estimator with generation parameters - %s", kwargs
        )

    @property
    def optimal_batch_size(self):
        """Property - optimal_batch_size"""
        return self._optimal_batch_size

    @property
    def model_generation_param_keys(self):
        """Property - model_generation_param_keys"""
        return self._model_generation_param_keys

    def fit(self, *args, **kwargs) -> BaseEstimator:  # pylint: disable=unused-argument
        """Dummy fit function which does not actually do anything :)

        Returns:
            self: returns self
        """
        self.is_fitted_ = True
        return self

    def predict(self, X: List) -> List:
        """Performs prediction on list of inputs `X`

        Args:
            X (List): list of inputs

        Returns:
            List: output produced by the model for each sample in `X`
        """
        # Gets the value of generation params using which prediction will run, when running any kind of search this is set via `.set_params`
        model_generation_params = {
            attr: getattr(self, attr) for attr in self._model_generation_param_keys
        }
        # Override inference function if present
        if self.pred_function:
            return self.pred_function(self, X, model_generation_params)
        # Perform inference on input data
        output, self._optimal_batch_size = infer_data(
            model=self.model,
            tokenizer=self.tokenizer,
            is_encoder_decoder=self.is_encoder_decoder,
            batch_size=self.batch_size,
            disable_batch_size_cache=self.disable_batch_size_cache,
            device=self.device,
            model_inputs=X,
            tokenizer_encoding_kwargs=self.tokenizer_encoding_kwargs,
            generation_kwargs=model_generation_params,
            disable_generation_param_checks=self.disable_generation_param_checks,
            tokenizer_decoding_kwargs=self.tokenizer_decoding_kwargs,
            return_optimal_batch_size=True,
        )
        return output

    def __sklearn_clone__(self) -> BaseEstimator:
        """Returns the same estimator
        Only relevant from sklearn >= 1.3

        - This functions returns the same estimator, as there are no parameters to specifically "fit"
        - This is done to avoid OOM errors for larger models and prevent creating a copy of the model weights, this does not affect the hyperparameter search in any way

        Returns:
            BaseEstimator: self
        """
        _ = self._get_model_generation_params()
        logger.debug("Attributes before cloning estimator - %s", _)
        return self

    def get_params(self, deep: bool = True):
        """
        Gets all of the parameters defined when calling `__init__` + generation parameters if set

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
        out = {}
        for key, value in vars(self).items():
            # Ignore any private/protected variables
            if not (key.startswith("__") or key.startswith("_")):
                out[key] = value
        return out

    def set_params(self, **params):
        """Sets the generation parameters of this estimator

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

        # set the keys each time this method is called
        self._model_generation_param_keys = list(params.keys())

        for key, value in params.items():
            setattr(self, key, value)

        _ = self._get_model_generation_params()
        logger.debug("Attributes after setting new parameters - %s", _)
        return self

    def _get_model_generation_params(self):
        model_generation_params = {
            attr: getattr(self, attr) for attr in self._model_generation_param_keys
        }
        return model_generation_params


class Tuner:
    """Tuner Class which drives the search for the generation hyperparameters"""

    def __init__(
        self,
        model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM],
        tokenizer: AutoTokenizer,
        prompt_template: langchain.BasePromptTemplate,
        dataset: Union[Dataset, Dict],
        column_mapping: Dict,
        device: str,
        is_encoder_decoder: bool,
        scorer: Callable,
        greater_is_better: bool = True,
        seed: int = 42,
        tokenizer_encoding_kwargs: Dict = None,
        tokenizer_decoding_kwargs: Dict = None,
        batch_size: int = 32,
        disable_batch_size_cache: bool = False,
        disable_generation_param_checks: bool = False,
        sample_ratio: float = 0.3,
        tokenizer_max_length_quantile: float = 0.9,
    ):
        """Tuner Class

        Args:
            model (Union[AutoModelForCausalLM,AutoModelForSeq2SeqLM]): model that has `.generate` method
            tokenizer (AutoTokenizer): tokenizer for the input
            prompt_template (langchain.BasePromptTemplate): prompt template that will apply to the input(`X`) of the model
            dataset (Union[Dataset, Dict]): The dataset, The processed version(after applying the prompt template) of the input and output are stored with key `X` & `y` in `dataset` of the object
            column_mapping (Dict): A mapping from the column names in the `dataset` to the column names expected by the model. The expected format is a dictionary with the following format: {"text_column_name": "X", "label_column_name": "y"}.
            device (str): device to run inference on, eg - `cuda:0`
            is_encoder_decoder (bool): whether the model is an encoder-decoder model, `False` if not
            scorer (Callable): A function that has this signature - `(y_true: List, y_pred: List) -> float` , takes in ground truth and predictions are returns a metric to optimize on
            greater_is_better (bool, optional): scorer is a scoring(greater is better) or a loss function(lower is better) . Defaults to True.
            seed (int, optional): seed for reproducibility, Only used for computing tokenizer arguments in `self.get_default_input_tokenizer_kwargs`.  Defaults to 42.
            tokenizer_encoding_kwargs (Dict, optional): Encoding arguments for the `tokenizer`. If `None` it's initialized using the `get_default_input_tokenizer_kwargs` method. Defaults to None.
            tokenizer_decoding_kwargs (Dict, optional): Decoding arguments for the `tokenizer`. Defaults to `{'skip_special_tokens' : True}`.
            batch_size (int, optional): batch_size to run inference with, this gets dynamically halfed if the inference function encounters OOM errors. Defaults to 32.
            disable_batch_size_cache (bool, optional): If `True` for each cross validation run, the pre-defined `batch_size` is used, this could lead to wasted computation time if OOM is raised by the inference function. Defaults to False.
            disable_generation_param_checks (bool, optional): Disables the custom generation parameter checks, this check does a sanity check of the parameters & produces warnings before doing generation, Not stable right now.  Defaults to False.
            sample_ratio (float, optional): Sampling Ratio of `dataset` to find the ideal values for padding and truncation to batch inputs to the model. Argument is invalid if `tokenizer_encoding_kwargs` is not `None`. Defaults to 0.3.
            tokenizer_max_length_quantile (float, optional): quantile at which the value for `max_length` will be computed using the initialized dataset. Defaults to 0.9.
        """
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.column_mapping = column_mapping
        # Map prompt template to dataset
        self.dataset = dataset.map(
            lambda sample: {
                "X": self.prompt_template.format(
                    X=sample[column_mapping["text_column_name"]]
                ),
                "y": sample[column_mapping["label_column_name"]],
            }
        )
        self.device = device
        self.score_func = scorer
        # Make scorer
        self.scorer = make_scorer(
            score_func=scorer, greater_is_better=greater_is_better
        )
        self.seed = seed
        self.disable_batch_size_cache = disable_batch_size_cache
        self.disable_generation_param_checks = disable_generation_param_checks
        self.sample_ratio = sample_ratio
        self.tokenizer_max_length_quantile = tokenizer_max_length_quantile
        # Get encoding arguments for the `tokenizer` if not passed in
        self.tokenizer_encoding_kwargs = self.get_default_input_tokenizer_kwargs(
            sample_ratio=sample_ratio,
            tokenizer_kwargs=tokenizer_encoding_kwargs,
        )
        self.tokenizer_decoding_kwargs = (
            tokenizer_decoding_kwargs
            if tokenizer_decoding_kwargs
            else {"skip_special_tokens": True}
        )
        # Initialize the model estimator
        self.estimator = LLMEstimatorWrapper(
            model=model,
            tokenizer=self.tokenizer,
            is_encoder_decoder=is_encoder_decoder,
            device=self.device,
            scorer=self.scorer,
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
        """Get default input tokenizer arguments using the dataset, Sets `padding` & `truncation` to True and calculate `max_length` as tokenizer arguments

        Args:
            sample_ratio (float): Sampling Ratio of `dataset` to find the ideal values for padding and truncation. Argument is invalid if `tokenizer_encoding_kwargs` is not `None`.
            tokenizer_length_percentile (float): percentile to find a value for `max_length` based on the dataset
            tokenizer_kwargs (Union[Dict, None]): Encoding key value arguments for the `tokenizer`. Returns the same value if this is not `None`. Defaults to None.
        """

        if tokenizer_kwargs:
            return tokenizer_kwargs

        tokenizer_encoding_kwargs = {
            "padding": True,
            "truncation": True,
        }
        logger.info("Computing tokenizer encoding arguments using a sample of the dataset...")
        sample_size = int(len(self.dataset["y"]) * sample_ratio)
        random.seed(self.seed)
        sample_indexes = random.sample(range(0, len(self.dataset["y"])), sample_size)
        get_items = itemgetter(*sample_indexes)
        X = get_items(self.dataset["X"])
        # Calculate max_length
        max_length_at_quantile = self.get_value_at_quantile(
            input_list=X, quantile=self.tokenizer_max_length_quantile
        )
        tokenizer_encoding_kwargs['max_length'] = min(max_length_at_quantile, self.tokenizer.model_max_length)
        logger.debug("Computed max length at quantile - %d, tokenizer model max length - %d", max_length_at_quantile, tokenizer_encoding_kwargs['max_length'])
        logger.info(
            "Setting tokenizer encoding arguments to - %s", tokenizer_encoding_kwargs
        )
        return tokenizer_encoding_kwargs

    def get_score(
        self, generation_kwargs: Dict, dataset: Union[Dataset, Dict] = None
    ) -> Tuple[float, List]:
        """Evaluate the score function on a dataset or the initialized dataset using some generation arguments for the model

        Args:
            generation_kwargs (Dict): generation kwargs to perform inference
            dataset (Union[Dataset, Dict], optional): dataset to perform inference on. Defaults to None.

        Returns:
            Tuple[float, List]: score, predictions
        """
        dataset_to_evaluate = dataset if dataset else self.dataset
        y_true = dataset_to_evaluate["y"]
        y_pred = infer_data(
            model=self.estimator.model,
            tokenizer=self.tokenizer,
            is_encoder_decoder=self.estimator.is_encoder_decoder,
            batch_size=self.estimator.optimal_batch_size,
            device=self.device,
            model_inputs=dataset_to_evaluate["X"],
            tokenizer_encoding_kwargs=self.tokenizer_encoding_kwargs,
            tokenizer_decoding_kwargs=self.tokenizer_decoding_kwargs,
            generation_kwargs=generation_kwargs,
            disable_batch_size_cache=self.disable_batch_size_cache,
            disable_generation_param_checks=self.disable_generation_param_checks,
        )
        score = self.score_func(y_true=y_true, y_pred=y_pred)
        return score, y_pred

    def get_value_at_quantile(self, input_list: List, quantile: float) -> int:
        """Get value at a specific quantile

        Args:
            input_list (List): list of str on which to run the encoding of the `tokenizer`
            quantile (float): quantile on which to find the value on.

        Returns:
            int: rounded value at quantile
        """
        # we don't consider `self.tokenizer.model_max_length` as the objective is to get a sense of the input
        input_ids = self.tokenizer(
            input_list, max_length=None, truncation=False, padding=False
        )["input_ids"]
        batch_ids = list(map(len, input_ids))
        return round(np.quantile(batch_ids, q=quantile))
