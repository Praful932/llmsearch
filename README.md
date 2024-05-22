![llmsearch](assets/llmsearch.png)
# llmsearch

A familiar way to do  hyperparameter search over generation parameters of an LLM. All you need is a model, dataset and a metric.

# Installation
The package works best with `python>=3.8.1`, `torch>=1.1` and `transformers>=4.27.4`.
```
pip install llmsearch
```

## Getting Started

### QuickStart
- [llama-3-8b Example]() - Snippet which shows generation parameter search.

### End-to-End Model Examples
1. [GSM8K Example](https://github.com/Praful932/llmsearch/blob/main/examples/gsm8k_example.ipynb) - Shows a GridSearch ran on the [GSM8K](https://huggingface.co/datasets/gsm8k) Dataset using the `TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ` model.
2. [Samsum Example](https://github.com/Praful932/llmsearch/blob/main/examples/samsum_example.ipynb) - Shows a GridSearch ran on the samsum Dataset
using a finetuned(on samsum dataset) version of `cognitivecomputations/dolphin-2.2.1-mistral-7b`.

### Snippets
<details>
  <summary>Instantiate a Tuner object</summary>

```python
# Requires accelerate==0.27.2 py7zr==0.21.0 evaluate==0.4.0 rouge_score==0.1.2

import torch
import evaluate
import datasets
import numpy as np

from llmsearch.tuner import Tuner
from llmsearch.scripts.stopping_criteria import MultiTokenStoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList

seed = 42
batch_size = 2
num_samples = 10

# load model & tokenizer
model_id = "cognitivecomputations/dolphin-2.9-llama3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = "left")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.float16, device_map = "auto")

# load dataset on which to run search on
dataset = datasets.load_dataset("samsum")['train']
sample_dataset = dataset.shuffle(seed = seed).select(range(num_samples))


# Optional : Define stopping criteria for the generation, here we stop a generation of a sequence when `<|im_end|>` is reached
multi_token_stop_criteria_ob = MultiTokenStoppingCriteria(sequence_ids=[128256])
stopping_criteria = StoppingCriteriaList([multi_token_stop_criteria_ob])
# useful when batching to reset state variables for the stopping criteria
callbacks_after_inference = [multi_token_stop_criteria_ob.reset]

# create a function that can be useful for evaluation
rouge = evaluate.load('rouge')
def get_rouge_score(y_true, y_pred):
    return np.mean(rouge.compute(predictions=y_pred, references=[item['summary'] for item in y_true], use_stemmer=True, use_aggregator=False)['rouge2'])

# Define a dataset preprocessor - Should take in tokenizer & kwargs and return a string that can be input directly to the model, here we apply chat template which most decoder models use
def sample_to_chat_format(tokenizer, **kwargs):
    messages = [
        {
            'role' : "system",
            'content' : "You are Dolphin, a helpful AI assistant."
        },
        {
            'role' : "user",
            'content' : f"Summarize the following text: {kwargs['dialogue']}"
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

# define tuner object, this preprocesses the dataset and creates an LLMEstimator to run with scikit-learn
tuner_ob = Tuner(
    model=model,
    tokenizer=tokenizer,
    dataset=sample_dataset,
    device="cuda:0",
    # the tuner module automatically reduces the batch size while running inference if it goes OOM
    batch_size=batch_size,
    tokenizer_encode_args={"padding": "longest",'truncation' : True, "add_special_tokens": False, 'max_length' : 1024},
    tokenizer_decode_args={"spaces_between_special_tokens": False, 'skip_special_tokens' : True},
    # pass in the scorer
    scorer=get_rouge_score,
    # pass in `dataset` preprocessor
    sample_preprocessor=sample_to_chat_format,
    seed=seed,
    # column mapping used to identify input and evaluation columns (these columns are passed in to the evaluation function & the dataset preprocessor)
    column_mapping={"input_cols": ["dialogue"], "eval_cols": ["summary"]},
    # callbacks if any to run after each inference
    callbacks_after_inference=callbacks_after_inference,
)

# Check to see if dataset is processed as expected, tuner_ob populates `_X` with the processed input and `_y` with `column_mapping.eval_cols`

```
</details>
<details>
  <summary>Inspect the Tuner Object</summary>

```python
# Assuming tuner_ob is built


```
</details>



## Recommendations
1. Searching for generation parameters are generally useful for tasks which have a generative factor involved - output is a variable length text and is not constrained to certain discrete outputs. For instance, searching for generation parameters would be more valuable in a task like summarization than classification where the output is constrained.
2. Batch size can affect performance during evaluation, most decoder models use `left` padding, The presence of pad token can affect the next token is generated although very minutely, the effect becomes more pronounced over long sequences. So be sure to evaluate the right batch size settings for your specific use case.
3. Use a stopping criteria while evaluating models so that model does not endlessly generate tokens until `max_new_tokens` is reached. All of the examples in the repo use a [stopping criteria]().

## Reproducibility
- Running a hyperparameter search works best when results are reproducible (you get similar outputs across different runs).
- There are quantization frameworks such as `exllama` offer very high inference speed while trading against reproducibility. `AWQ` is one of those quantization methods that is fast and also offers reproducibility (This has been used in one of the examples showed above). Note that there may be usecase that may work well even with `exllama` - problems where there is a soft spot for certain generation parameters.
- For the generation to be reproducible there is a `generation_seed` parameter that has been introduced which is added to the `transformers` module via monkey patching. This is used while running inference to seed outputs while running `model.generate`. This can be also treated as a hyperparameter in itself.
- Batch size can affect performance during evaluation, most decoder models use `left` padding, The presence of pad token can affect the next token samples that is generated although very minutely, the effect becomes more pronounced over long sequences. So be sure to evaluate your model at the right batch size.

## Important Considerations
- `llmsearch` monkey patches certain modules of transformers to make it work with `sklearn` specifically these ones
    - Changed Generation Related Modules - This is required to add support for certain generation stratergies that are not natively supported in HF, specifically - `tfs` & `top_a`, It also adds one other param called `generation_seed` which is used for seeding the generation for reproducibilty
        - `transformers.GenerationMixin._get_logits_warper` - Older module available at `transformers.GenerationMixin._get_logits_warper_old`
        - `transformers.GenerationConfig.__init__` - Older constructor available via `transformers.GenerationConfig.__init__`
    - Added a new attribute to `StoppingCriteriaList` class which helps in avoiding cloning of the same object while running a search, which otherwise would have destroyed the state of the object
        - `StoppingCriteriaList.__sklearn_clone__`




# Benchmarks



# Contents
- Why?
- Usage
- Benchmarks
- References

### Usage