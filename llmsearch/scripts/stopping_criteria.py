"""Stopping criteria utilities for generation."""

from typing import List

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

class MultiTokenStoppingCriteria(StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence.

    A modified verion of Stopping Criteria implemented here - https://github.com/EleutherAI/lm-evaluation-harness/blob/27924d77953491f66a038a09892807065e469358/lm_eval/models/utils.py#L208
    That maintains a state for each batch of inputs which helps in knowing where to look.

    This code is not thread safe. The same object cannot be used simultaneously in multiple threads.
    """

    def __init__(
        self,
        sequence_ids : List[int],
    ) -> None:
        self.sequence_ids = torch.tensor(sequence_ids, dtype = torch.int32, device = "cuda:0")
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization
        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = self.sequence_ids.shape[0] + 2
        self.state_initialized = False
        self.prompt_length = None
        self.state_initialized = False

        self.batch_size, self.done_tracker = None, []

    def set_state(self, batch_size, prompt_length):
        self.batch_size = batch_size
        self.prompt_length = prompt_length
        self.done_tracker = [False] * batch_size
        self.state_initialized = True

    def reset(self):
        self.batch_size = None
        self.prompt_length = None
        self.state_initialized = False


    def __call__(self, input_ids, scores, **kwargs) -> bool:
        """
        This is called after a new token is generated
        """
        ret_val = False

        if not self.state_initialized:
            # Every batch should set this state
            self.set_state(input_ids.shape[0], input_ids.shape[1] - 1)

        # IDs of all the tokens except the prompt
        lookback_ids_batch = input_ids[:, self.prompt_length :]
        # look back for 2 more tokens than it takes to encode our stop sequence
        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        # no elements yet to look back
        if lookback_ids_batch.nelement() == 0:
            return False

        for i, done in enumerate(self.done_tracker):
            if not done:
                # look back as much as the length of the stop token sequence
                self.done_tracker[i] = self.sequence_ids == lookback_ids_batch[i][-(self.sequence_ids.shape[0]):]
        ret_val = False not in self.done_tracker
        return ret_val
