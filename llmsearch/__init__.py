"""

<a href="https://freeimage.host/"><img src="https://iili.io/JpEgiIS.png" alt="JpEgiIS.png" border="0" /></a>

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Praful932/llmsearch)
[![Netlify Status](https://api.netlify.com/api/v1/badges/90e52247-da13-447b-b2eb-8f169a514877/deploy-status)](https://app.netlify.com/sites/llmsearch/deploys)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPraful932%2Fllmsearch&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


Conduct hyperparameter search over generation parameters of large language models (LLMs). This tool is designed for ML practitioners looking to optimize their sampling strategies to improve model performance. Simply provide a model, dataset, and performance metric, llmsearch handles the rest.

### QuickStart
-  llama-3-8b Example [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Praful932/llmsearch/blob/main/examples/llmsearch_quickstart.ipynb) - A quickstart notebook which shows the basic functionality of llmsearch. This notebook will help you understand how to quickly set up and run hyperparameter searches.

### End-to-End Model Examples
1. [GSM8K Example](https://github.com/Praful932/llmsearch/blob/main/examples/gsm8k_example.ipynb) - Shows a `GridSearchCV` ran on the [GSM8K](https://huggingface.co/datasets/gsm8k) Dataset using the `TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ` model.
2. [Samsum Example](https://github.com/Praful932/llmsearch/blob/main/examples/samsum_example.ipynb) - Shows a `GridSearchCV` ran on the [samsum](https://huggingface.co/datasets/samsum) Dataset
using a finetuned(on the same dataset) version of `cognitivecomputations/dolphin-2.2.1-mistral-7b`.

Refer [README](https://github.com/Praful932/llmsearch) further for more details on how to use `llmsearch`.

# API Reference Documentation
"""
from llmsearch.patches.transformers_monkey_patch import hijack_samplers

print("Monkey Patching .generate function of `transformers` library")
hijack_samplers()

__version__ = "0.1.0"