"""

<a href="https://freeimage.host/"><img src="https://iili.io/JpEgiIS.png" alt="JpEgiIS.png" border="0" /></a>

Conduct hyperparameter search over generation parameters of large language models (LLMs). This tool is designed for ML practitioners looking to optimize their sampling strategies to improve model performance. Simply provide a model, dataset, and performance metric, llmsearch handles the rest.

### QuickStart
-  llama-3-8b Example [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Praful932/llmsearch/blob/main/examples/llmsearch_quickstart.ipynb) - A quickstart notebook which shows the basic functionality of llmsearch. This notebook will help you understand how to quickly set up and run hyperparameter searches.

### End-to-End Model Examples
1. [GSM8K Example](https://github.com/Praful932/llmsearch/blob/main/examples/gsm8k_example.ipynb) - Shows a `GridSearchCV` ran on the [GSM8K](https://huggingface.co/datasets/gsm8k) Dataset using the `TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ` model.
2. [Samsum Example](https://github.com/Praful932/llmsearch/blob/main/examples/samsum_example.ipynb) - Shows a `GridSearchCV` ran on the [samsum](https://huggingface.co/datasets/samsum) Dataset
using a finetuned(on the same dataset) version of `cognitivecomputations/dolphin-2.2.1-mistral-7b`.

Refer [README](https://github.com/Praful932/llmsearch) for more details on how to use llmsearch.

# API Reference Documentation
"""
from llmsearch.patches.transformers_monkey_patch import hijack_samplers

print("Monkey Patching .generate function of `transformers` library")
hijack_samplers()

__version__ = "0.1.0"