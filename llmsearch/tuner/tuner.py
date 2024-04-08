"""
Main Tuner Module containing LLMEstimatorWrapper Class & Tuner Class for scikit-learn
"""

import random
from operator import itemgetter
from typing import List, Union, Tuple, Dict, Callable

import numpy as np
from torch import nn
from datasets import Dataset
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from transformers import AutoTokenizer

from llmsearch.utils.model_utils import run_inference
from llmsearch.utils.mem_utils import cache

from llmsearch.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMEstimatorWrapper(BaseEstimator):
    """Estimator Wrapper Class that abstracts a Model for compatibility with scikit-learn"""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        scorer: Callable,
        device: str,
        tokenizer_encode_args: Dict,
        tokenizer_decode_args: Dict,
        batch_size: int,
        callbacks_after_inference: List[Callable] = None,
        disable_batch_size_cache: bool = False,
        pred_function: Union[Callable, None] = None,
        is_encoder_decoder: bool = False,
        disable_generation_param_checks: bool = False,
        output_preproc: Callable = lambda x: x.strip(),
        **kwargs,
    ):
        """Initializes the Estimator

        Args:
            model (nn.Module): model that has a `.generate` method
            tokenizer (AutoTokenizer): tokenizer for the input
            scorer (Callable): A function that has this signature - `(y_true: List, y_pred: List) -> float` , takes in ground truth and predictions are returns a metric to optimize on
            device (str): device to run inference on, eg - `cuda:0`
            tokenizer_encode_args (Dict): Encoding key value arguments for the `tokenizer`
            tokenizer_decode_args (Dict): Decoding key value arguments for the `tokenizer`
            batch_size (int): batch_size to run inference with, this gets dynamically halfed if the inference function encounters OOM errors
            callbacks_after_inference (List[Callable], optional): Callbacks to run after each inference. Useful for stopping criteria in generation. Defaults to None.
            disable_batch_size_cache (bool): If `True` for each cross validation run, the pre-defined `batch_size` is used, this could lead to wasted computation time if OOM is raised by the inference function, Defaults to False.
            pred_function (Union[Callable, None], optional): Override inference function if present. Defaults to None.
            is_encoder_decoder (bool, optional): whether the model is an encoder-decoder model, `False` if not. Defaults to False.
            disable_generation_param_checks (bool, optional): Disables the custom generation parameter checks, this does a sanity check of the parameters & produces warnings before doing generation. Defaults to False.
            output_preproc (Callable, optional): Post processing function for the output, by default it strips the output. Defaults to `lambda x : x.strip()`.

        - All other kwargs are assumed to be generation parameters and used when doing hyperparam search
        """
        self.model = model
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.device = device
        self.tokenizer_encode_args = tokenizer_encode_args
        self.tokenizer_decode_args = tokenizer_decode_args
        self.batch_size = batch_size
        # stores the optimal batch size for a particular configuration if disable_batch_size_cache is `False`
        self._optimal_batch_size = batch_size
        self.callbacks_after_inference = [] if callbacks_after_inference is None else callbacks_after_inference
        self.disable_batch_size_cache = disable_batch_size_cache
        self.pred_function = pred_function
        self.is_encoder_decoder = is_encoder_decoder
        self.disable_generation_param_checks = disable_generation_param_checks
        self.output_preproc = output_preproc

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
        output, self._optimal_batch_size = run_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            is_encoder_decoder=self.is_encoder_decoder,
            batch_size=self.batch_size,
            disable_batch_size_cache=self.disable_batch_size_cache,
            device=self.device,
            model_inputs=X,
            tokenizer_encode_args=self.tokenizer_encode_args,
            generation_args=model_generation_params,
            disable_generation_param_checks=self.disable_generation_param_checks,
            tokenizer_decode_args=self.tokenizer_decode_args,
            return_optimal_batch_size=True,
            callbacks=self.callbacks_after_inference,
            output_preproc=self.output_preproc,
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
        model: nn.Module,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        column_mapping: Dict[str, list],
        prompt_template: str,
        scorer: Callable,
        device: str,
        tokenizer_encode_args: Dict = None,
        tokenizer_decode_args: Dict = None,
        batch_size: int = 16,
        output_preproc: Callable = lambda x: x.strip(),
        callbacks_after_inference: List[Callable] = None,
        is_encoder_decoder: bool = False,
        greater_is_better: bool = True,
        seed: int = 42,
        disable_batch_size_cache: bool = False,
        disable_generation_param_checks: bool = False,
        sample_ratio: float = 0.3,
        tokenizer_max_length_quantile: float = 0.9,
        custom_pred_function = None,
    ):
        """Tuner Class

        Args:
            model (nn.Module): model that has a `.generate` method
            tokenizer (AutoTokenizer): tokenizer for the input
            dataset (Dataset): dataset to perform search on
            column_mapping (Dict[str, list]): should contain `input_cols` & `eval_cols` keys, `input_cols` should contain the columns to be used in the `prompt_template` & `eval_cols` should contain the columns to be used in the `scorer`,
                all eval_columns will be passed in as a dict as the second argument to the `scorer` function, eg - `{'input_cols' : ["question"], 'eval_cols' : ['answer']}`
            prompt_template (str): template for the prompt, should contain the keys from `input_cols` in the `column_mapping` eg prompt template - `"Question : How many days are there in a year?\nAnswer : 365\n\nQuestion : {question}\nAnswer : "`
            scorer (Callable): A function that has this signature - `(y_true: List, y_pred: List) -> float` , takes in ground truth and predictions are returns a metric to optimize on, `eval_cols` in `column_mapping` are passed in as the second argument as a List[Dict]
            device (str): device to run inference on, eg - `cuda:0`
            tokenizer_encode_args (Dict, optional): Encoding key value arguments for the `tokenizer`. If `None` it's initialized using the `get_default_input_tokenizer_kwargs` method. Defaults to None.
            tokenizer_decode_args (Dict, optional): Decoding key value arguments for the `tokenizer`. Defaults to `{'skip_special_tokens' : True}`.
            batch_size (int, optional): batch_size to run inference with, this gets dynamically halfed if the inference function encounters OOM errors. Defaults to 16.
            output_preproc (Callable, optional): Post processing function for the output, by default it strips the output. Defaults to `lambda x : x.strip()`.
            callbacks_after_inference (List[Callable], optional): Callbacks to run after each inference. Useful for stopping criteria in generation. Defaults to None.
            is_encoder_decoder (bool, optional): whether the model is an encoder-decoder model, `False` if not. Defaults to False.
            greater_is_better (bool, optional): whether the metric to optimize on is greater the better. Defaults to True.
            seed (int, optional): seed for reproducibility. Defaults to 42.
            disable_batch_size_cache (bool, optional): If `True` for each cross validation run, the pre-defined `batch_size` is used, this could lead to wasted computation time if OOM is raised by the inference function. Defaults to False.
            disable_generation_param_checks (bool, optional): Disables the custom generation parameter checks, this does a sanity check of the parameters & produces warnings before doing generation. Defaults to False.
            custom_pred_function (Union[Callable, None], optional): Override inference function if present. Defaults to None. Should take in two parameters - model inputs (List[str]) and model generation parameters (Dict) and return a List[str] of outputs
                overrides model_utils.run_inference

            # TODO : To check
            sample_ratio (float, optional): Sampling Ratio of `dataset` to find the ideal values for padding and truncation. Argument is invalid if `tokenizer_encode_args` is not `None`. Defaults to 0.3.
            tokenizer_max_length_quantile (float, optional): percentile to find a value for `max_length` based on the dataset. Defaults to 0.9.
        """
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.column_mapping = column_mapping
        self.input_cols = column_mapping["input_cols"]
        self.eval_cols = column_mapping["eval_cols"]

        self.dataset = dataset.map(
            lambda sample: {
                "_X": self.prompt_template.format(
                    **{input_col: sample[input_col] for input_col in self.input_cols}
                ),
                "_y": {eval_col: sample[eval_col] for eval_col in self.eval_cols},
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

        # TODO - to check
        self.sample_ratio = sample_ratio
        self.tokenizer_max_length_quantile = tokenizer_max_length_quantile

        # Get encoding arguments for the `tokenizer` if not passed in
        self.tokenizer_encode_args = self.get_default_tokenizer_encode_args(
            sample_ratio=sample_ratio,
            tokenizer_encode_args=tokenizer_encode_args,
        )
        self.tokenizer_decode_args = (
            tokenizer_decode_args
            if tokenizer_decode_args
            else {"skip_special_tokens": True}
        )
        # Initialize the model estimator
        self.estimator = LLMEstimatorWrapper(
            model=model,
            tokenizer=self.tokenizer,
            scorer=self.scorer,
            device=self.device,
            tokenizer_encode_args=self.tokenizer_encode_args,
            tokenizer_decode_args=self.tokenizer_decode_args,
            batch_size=batch_size,
            callbacks_after_inference=callbacks_after_inference,
            disable_batch_size_cache=disable_batch_size_cache,
            pred_function=custom_pred_function,
            is_encoder_decoder=is_encoder_decoder,
            disable_generation_param_checks=disable_generation_param_checks,
            output_preproc=output_preproc,
        )
        # reset cache on every init
        cache.empty_cache()

    def get_default_tokenizer_encode_args(
        self,
        sample_ratio: float,
        tokenizer_encode_args: Union[Dict, None],
    ):
        """Get default input tokenizer arguments using the dataset, Sets `padding` & `truncation` to True and calculate `max_length` as tokenizer arguments

        Args:
            sample_ratio (float): Sampling Ratio of `dataset` to find the ideal values for padding and truncation. Argument is invalid if `tokenizer_encode_args` is not `None`.
            tokenizer_length_percentile (float): percentile to find a value for `max_length` based on the dataset
            tokenizer_kwargs (Union[Dict, None]): Encoding key value arguments for the `tokenizer`. Returns the same value if this is not `None`. Defaults to None.
        """
        # TODO - Check functionality of this method
        _quantiles_to_calc = [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.97, 0.99]

        if tokenizer_encode_args:
            return tokenizer_encode_args

        tokenizer_encode_args = {
            "padding": 'longest',
            "truncation": True,
            'add_special_tokens' : False,
        }
        logger.info(
            "Computing tokenizer encoding arguments using a sample of the dataset..."
        )
        sample_size = int(len(self.dataset["_X"]) * sample_ratio)
        random.seed(self.seed)
        sample_indexes = random.sample(range(0, len(self.dataset["_X"])), sample_size)
        get_items = itemgetter(*sample_indexes)
        X = get_items(self.dataset["_X"])
        # Calculate max_length
        max_length_at_quantile = self.get_value_at_quantile(
            input_list=X, quantile=
            self.tokenizer_max_length_quantile
        )

        # TODO : will max length and above config work?
        tokenizer_encode_args["max_length"] = min(
            max_length_at_quantile, self.tokenizer.model_max_length
        )
        logger.debug(
            "Computed max length at quantile - %d, tokenizer model max length - %d",
            max_length_at_quantile,
            tokenizer_encode_args["max_length"],
        )

        logger.info(
            "Setting tokenizer encoding arguments to - %s", tokenizer_encode_args
        )
        return tokenizer_encode_args

    def get_score(
        self, generation_args: Dict, dataset: Union[Dataset, Dict] = None
    ) -> Tuple[float, List]:
        """Evaluate the score function on a dataset or the initialized dataset using some generation arguments for the model

        Args:
            generation_args (Dict): generation kwargs to perform inference
            dataset (Union[Dataset, Dict], optional): dataset to perform inference on. Defaults to None.

        Returns:
            Tuple[float, List]: score, predictions
        """
        dataset_to_evaluate = dataset if dataset else self.dataset
        if not dataset:
            dataset_to_evaluate = self.dataset
        else:
            dataset_to_evaluate = dataset.map(
                lambda sample: {
                    "_X": self.prompt_template.format(
                        **{input_col: sample[input_col] for input_col in self.input_cols}
                    ),
                    "_y": {eval_col: sample[eval_col] for eval_col in self.eval_cols},
                }
            )
        y_pred = run_inference(
            model=self.estimator.model,
            tokenizer=self.estimator.tokenizer,
            is_encoder_decoder=self.estimator.is_encoder_decoder,
            batch_size=self.estimator.optimal_batch_size,
            device=self.device,
            model_inputs=dataset_to_evaluate["_X"],
            tokenizer_encode_args=self.tokenizer_encode_args,
            tokenizer_decode_args=self.tokenizer_decode_args,
            generation_args=generation_args,
            disable_batch_size_cache=self.disable_batch_size_cache,
            disable_generation_param_checks=self.disable_generation_param_checks,
            output_preproc=self.estimator.output_preproc,
            callbacks=self.estimator.callbacks_after_inference,
        )
        score = self.score_func(y_true=self.dataset["_y"], y_pred=y_pred)
        return score, y_pred

    def get_value_at_quantile(self, input_list: List, quantile: float) -> int:
        """Get value at a specific quantile

        Args:
            input_list (List): list of str on which to run the encoding of the `tokenizer`
            quantile (float): quantile on which to find the value on.

        Returns:
            int: rounded value at quantile
        """
        input_ids = self.tokenizer(
            input_list, max_length=None, truncation=False, padding=False
        )["input_ids"]
        batch_ids = list(map(len, input_ids))
        return round(np.quantile(batch_ids, q=quantile))
