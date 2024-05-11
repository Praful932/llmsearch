![llmsearch](assets/llmsearch.png)
# llmsearch

This framework provides a familiar way to do a hyperparameter search over the generation parameters to find the best ones for your usecase.  All you need is a model, dataset and a metric.

# Installation
The package works best with `python>=3.8.1`, `torch>=1.1` and `transformers>=4.27.4`.

Install with pip
```
pip install llmsearch
```

# Getting Started

## End-to-End Model Examples
1. GSM8K Example - Shows a GridSearch ran on the GSM8K Dataset using the `TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ` model.
2. Samsum Example - Shows a GridSearch ran on the samsum Dataset
using a finetuned(on samsum dataset) version of `cognitivecomputations/dolphin-2.2.1-mistral-7b`.

## QuickStart

```python

import datasets

from llmsearch.tuner import Tuner
from transformers import AutoTokenizer, AutoModel
from llmsearch.scripts.stopping_criteria import MultiTokenStoppingCriteria

model_id = "cognitivecomputations/dolphin-2.9-llama3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

dataset = datasets.load_dataset("samsum")['train']

```

## Recommendations

# Contents
- Why?
- Usage
- Benchmarks
- References

### Why?
Generation Parameters play a key role in generating output for a language model. The way the next token is sample is decided by these generation parameters. There are multiple different text generation strategies and you may not know which works best for your usecase. That's where llmsearch comes in.

### Usage