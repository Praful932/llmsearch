# pylint: skip-file
"""
Monkey Patch to add in generation params that are currently not supported by `transformers` library
Script Copied from - https://github.com/oobabooga/text-generation-webui/blob/main/modules/transformers_monkey_patch.py
"""
import math

import torch
import transformers
from transformers import LogitsWarper
from transformers.generation.logits_process import (
    LogitNormalization,
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
)

from llmsearch.utils.logging_utils import get_logger

logger = get_logger(__name__)


from transformers import AutoTokenizer

model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)


class TailFreeLogitsWarper(LogitsWarper):
    def __init__(
        self,
        tfs: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        tfs = float(tfs)
        if tfs < 0 or tfs > 1.0:
            raise ValueError(f"`tfs` has to be a float >= 0 and <= 1, but is {tfs}")
        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Compute second derivative normalized CDF
        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        # compute cdf from pdf
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # Remove tokens with CDF value above the threshold (token with 0 are kept)
        sorted_indices_to_remove = normalized_d2_cdf > self.tfs

        # Centre the distribution around the cutoff as in the original implementation of the algorithm
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopALogitsWarper(LogitsWarper):
    def __init__(
        self,
        top_a: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        top_a = float(top_a)
        if top_a < 0 or top_a > 1.0:
            raise ValueError(f"`top_a` has to be a float >= 0 and <= 1, but is {top_a}")
        self.top_a = top_a
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Remove tokens with probability less than top_a*(max(probs))^2 (token with 0 are kept)
        probs_max = probs[..., 0, None]
        sorted_indices_to_remove = probs < probs_max * probs_max * self.top_a

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class MirostatLogitsWarper(LogitsWarper):
    def __init__(
        self,
        mirostat_mode: int,
        mirostat_tau: float,
        mirostat_eta: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        print(f"here - {mirostat_mode}")
        if mirostat_mode not in [2]:
            raise ValueError(
                f"`mirostat` has to be a an integer 2, but is {mirostat_mode}"
            )
        self.mirostat_mode = mirostat_mode
        # learning rate
        self.mirostat_eta = mirostat_eta
        # target surprise value
        self.mirostat_tau = mirostat_tau
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        # max cross entropy/surprise
        self.mu = 2 * self.mirostat_tau
        self.batch_init_done = False

    def __call_(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        - current generation step is n
        - This method is called for each generation step, shape of inputs_ids starts with (batch_size, 1), then (batch_size, 2)
        it contains the previous token ID

        Args:
            input_ids (torch.LongTensor): (batch_size, step_idx) step_idx is the the current generation step
            scores (torch.FloatTensor): (batch_size, vocab_size) scores for the current token

        Returns:
            torch.FloatTensor: _description_
        """
        print(f"Iteration - {input_ids.shape[1]}")
        # input_ids = input_ids[0].reshape(-1, 1)
        # scores = scores[0].reshape(-1, 32128)
        if not self.batch_init_done:
            # Initialize state when first time the function is called
            self.mu = torch.full(size=(input_ids.shape[0],), fill_value=float(self.mu))
            self.batch_init_done = True

        # (batch_size, vocab_size), (batch_size, vocab_size)
        sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        print(sorted_indices.tolist()[0][:5])
        print(f"sorted logits tensor sum - {torch.sum(sorted_logits)}")
        # calculate probs from logits - (batch_size, vocab_size)
        prob_original = torch.softmax(sorted_logits, dim=-1).tolist()
        print(prob_original[0][:4])

        # Get candidate logits for each sample
        sorted_logits_batch = (
            []
        )  # (list of logits for each row, each list contains candidate logits fitting the criteria)
        # Go through each row's top probabilities
        j = None
        for i, row in enumerate(prob_original):
            # Truncate/discard the words with surprise values greater than mu
            # as you go below, surprise/cross entropy value increases
            row_logits = []
            for j, candidate in enumerate(row):
                if candidate > 0 and -math.log2(candidate) > self.mu[i]:
                    # If there is a single candidate only take that
                    if j == 0:
                        row_logits.append(sorted_logits[i][:1])
                    # If there is more than one, take all but this
                    else:
                        row_logits.append(sorted_logits[i][:j])
                    break
            row_logit_tensor = torch.cat(row_logits, dim=0)
            print(f"row logit tensor sum - {torch.sum(row_logit_tensor)}")
            sorted_logits_batch.append(row_logit_tensor)
        print(f"Break index i - {j}")
        prev_indices = []
        final_scores = []

        # iterate through candidates of each sample
        for i, row in enumerate(sorted_logits_batch):
            print(row.shape)
            softmaxed_candidates = torch.softmax(row.squeeze(0), dim=0)
            print(f"softmax-shape - {softmaxed_candidates.shape}")

            # pick a candidate index from all of the selected candidates
            prev_index = torch.multinomial(
                softmaxed_candidates, num_samples=1, replacement=True
            )
            print(f"previous index - {prev_index}")
            prev_indices.append(prev_index)

            print(f"logit val - {softmaxed_candidates[prev_index]}")
            observed_surprise_value = -math.log2(softmaxed_candidates[prev_index])
            print(f"obs surprise value - {observed_surprise_value}")

            # calculate error based on observed surprise value and target surprise value
            e = observed_surprise_value - self.mirostat_tau

            # update mu using learning rate and error

            self.mu[i] -= self.mirostat_eta * e

            # shape - (vocab_size,)
            sorted_indices_to_remove = torch.ones_like(scores[0], dtype=torch.bool)
            # set the token that is picked to False
            sorted_indices_to_remove[prev_index] = False

            # shape - (vocab_size,)
            # convert the mask to be able to map to the scores(unsorted version)
            indices_to_remove = sorted_indices_to_remove.unsqueeze(0).scatter(
                1, sorted_indices[i].unsqueeze(0), sorted_indices_to_remove.unsqueeze(0)
            )
            random_tensor = torch.arange(1, len(indices_to_remove[0]) + 1).to("mps")
            print(f"indices to remove - {sum(random_tensor* indices_to_remove[0])}")
            # mask all other tokens
            scores[i] = scores[i].masked_fill(indices_to_remove, self.filter_value)
            print(f"sum value - {torch.sum(scores[i][scores[i] >= 0])}")
        return scores

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # This happens step by step, each token is passed in, then logit is calculated
        # seems the first sample in the batch is getting decoded
        print(f"Iteration - {input_ids.shape[1]}")
        logits = scores[0]
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        print(sorted_indices.tolist()[:5])
        print(f"sorted logits tensor sum - {torch.sum(sorted_logits)}")
        prob_original = torch.softmax(sorted_logits, dim=-1).tolist()  # candidates
        print(prob_original[:4])

        # Truncate the words with surprise values greater than mu
        # as you go below, surprise/cross entropy value increases
        for i, candidate in enumerate(prob_original):
            if candidate > 0 and -math.log2(candidate) > self.mu:
                if i == 0:
                    sorted_logits = sorted_logits[:1]
                else:
                    sorted_logits = sorted_logits[:i]
                break
        print(f"row logit tensor sum - {torch.sum(sorted_logits)}")
        print(f"Break index i - {i}")

        # Normalize the probabilities of the remaining words
        prob_topk = torch.softmax(sorted_logits, dim=0)

        prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
        print(f"previous index - {prev_i}")

        print(f"logit val - {prob_topk[prev_i]}")
        observed_surprise = -math.log2(prob_topk[prev_i])
        print(f"obs surprise value - {observed_surprise}")
        self.e = observed_surprise - self.mirostat_tau

        # Update mu using the learning rate and error
        self.mu -= self.mirostat_eta * self.e

        sorted_indices_to_remove = torch.ones_like(scores[0], dtype=torch.bool)
        sorted_indices_to_remove[prev_i] = False

        indices_to_remove = sorted_indices_to_remove.unsqueeze(0).scatter(
            1, sorted_indices.unsqueeze(0), sorted_indices_to_remove.unsqueeze(0)
        )
        random_tensor = torch.arange(1, len(indices_to_remove[0]) + 1).to("mps")
        print(f"indices to remove - {sum(random_tensor* indices_to_remove[0])}")
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        print(f"sum value - {torch.sum(scores[0][scores[0] >= 0])}")
        return scores


class RepetitionPenaltyLogitsProcessorWithRange(LogitsProcessor):
    """
    Copied from the transformers library
    """

    def __init__(self, penalty: float, _range: int):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}"
            )

        self.penalty = penalty
        self._range = _range

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        input_ids = input_ids[:, -self._range :]
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, input_ids, score)
        return scores


def get_logits_warper_patch(self, generation_config):
    warpers = self._get_logits_warper_old(generation_config)
    warpers_to_add = LogitsProcessorList()
    min_tokens_to_keep = 2 if generation_config.num_beams > 1 else 1

    # only apply mirostat if mode is 2
    if (
        generation_config.mirostat_mode is not None
        and generation_config.mirostat_mode == 2
    ):
        warpers_to_add.append(
            MirostatLogitsWarper(
                mirostat_mode=generation_config.mirostat_mode,
                mirostat_eta=generation_config.mirostat_eta,
                mirostat_tau=generation_config.mirostat_tau,
                min_tokens_to_keep=min_tokens_to_keep,
            )
        )
        # We need to disable samplers other than temperature
        for warper in warpers:
            if not isinstance(warper, TemperatureLogitsWarper):
                warpers.remove(warper)
    else:
        print("here")
        if generation_config.tfs is not None and 0.0 <= generation_config.tfs <= 1.0:
            warpers_to_add.append(
                TailFreeLogitsWarper(
                    tfs=generation_config.tfs, min_tokens_to_keep=min_tokens_to_keep
                )
            )
        if (
            generation_config.top_a is not None
            and 0.0 <= generation_config.top_a <= 1.0
        ):
            warpers_to_add.append(
                TopALogitsWarper(
                    top_a=generation_config.top_a, min_tokens_to_keep=min_tokens_to_keep
                )
            )

    if warpers and isinstance(warpers[-1], LogitNormalization):
        warpers = warpers[:-1] + warpers_to_add + [warpers[-1]]
    else:
        warpers += warpers_to_add

    return warpers


def get_logits_processor_patch(self, **kwargs):
    result = self._get_logits_processor_old(**kwargs)
    repetition_penalty_range = kwargs["generation_config"].repetition_penalty_range
    repetition_penalty = kwargs["generation_config"].repetition_penalty

    # override `RepetitionPenaltyLogitsProcessor` with range version
    if repetition_penalty_range > 0:
        for i in range(len(result)):
            if result[i].__class__.__name__ == "RepetitionPenaltyLogitsProcessor":
                result[i] = RepetitionPenaltyLogitsProcessorWithRange(
                    repetition_penalty, repetition_penalty_range
                )
    return result


from llmsearch.utils.common_utils import print_call_stack


def generation_config_init_patch(self, **kwargs):
    self.__init___old(**kwargs)
    # Add in extra generation params
    self.tfs = kwargs.pop("tfs", None)
    self.top_a = kwargs.pop("top_a", 0.0)
    self.mirostat_mode = kwargs.pop("mirostat_mode", 0)
    self.mirostat_eta = kwargs.pop("mirostat_eta", 0.1)
    self.mirostat_tau = kwargs.pop("mirostat_tau", 5)
    self.repetition_penalty_range = kwargs.pop("repetition_penalty_range", 0)


def hijack_samplers():
    transformers.GenerationMixin._get_logits_warper_old = (
        transformers.GenerationMixin._get_logits_warper
    )
    transformers.GenerationMixin._get_logits_warper = get_logits_warper_patch

    transformers.GenerationMixin._get_logits_processor_old = (
        transformers.GenerationMixin._get_logits_processor
    )
    transformers.GenerationMixin._get_logits_processor = get_logits_processor_patch

    transformers.GenerationConfig.__init___old = transformers.GenerationConfig.__init__
    transformers.GenerationConfig.__init__ = generation_config_init_patch
