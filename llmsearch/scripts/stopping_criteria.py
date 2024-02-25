import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class SingleTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, token, num):
        super().__init__()
        self.token = token
        self.num = num

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        res = []

        # If input is a batch all items of the batch should satisfy the condition
        for item in input_ids:
            stop_count = (item == self.token).sum().item()
            decoded_text = tokenizer.decode(item, skip_special_tokens=True)
            if stop_count >= self.num:
                # print(decoded_text)
                res.append(True)
        return res


# stopping_criteria = StoppingCriteriaList([SingleTokenStoppingCriteria(token=tokenizer.encode(" END")[0], num = 5)])
