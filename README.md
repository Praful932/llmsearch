![llmsearch](assets/llmsearch.png)
# llmsearch

A familiar way to do  hyperparameter search over generation parameters of an LLM. All you need is a model, dataset and a metric.

## Contents
- [Installation](#installation)
- [Getting Started](#getting-started)
    - [QuickStart](#quickstart) - A notebook to get you started quickly
    - [End-to-End Model Examples](#end-to-end-model-examples) - Detailed Examples with specific models
- [Benchmarks](#benchmarks) - Performance comparisons of different models and configurations.
- [Recommendations](#recommendations) - Best practises
- [Reproducibility](#reproducibility) - Ensuring consistent and reproducible results.
- [Important Considerations](#important-considerations) - Key points to be aware of when using llmsearch
- [References](#references)
## Installation
The package works best with `python>=3.8.1`, `torch>=1.1` and `transformers>=4.27.4`.
```
pip install llmsearch
```

## Getting Started

### QuickStart
- [llama-3-8b Example]() - A Quickstart Notebook which shows basic functionality of llmsearch. This notebook will help you understand how to quickly set up and run hyperparameter searches.

### End-to-End Model Examples
1. [GSM8K Example](https://github.com/Praful932/llmsearch/blob/main/examples/gsm8k_example.ipynb) - Shows a `GridSearchCV` ran on the [GSM8K](https://huggingface.co/datasets/gsm8k) Dataset using the `TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ` model.
2. [Samsum Example](https://github.com/Praful932/llmsearch/blob/main/examples/samsum_example.ipynb) - Shows a `GridSearchCV` ran on the [samsum]() Dataset
using a finetuned(on the same dataset) version of `cognitivecomputations/dolphin-2.2.1-mistral-7b`.

## Benchmarks

![llmsearch](assets/bm_gsm8k.png)
![llmsearch](assets/bm_samsum.png)

Table shows final metrics on out-of-sample corpus as a result of the search from the e2e examples above

| Model                                                   | Dataset | Before  | After   | Samples | Metric    | Best Parameters                                                                                                                                                     | Metric File                                            |
|---------------------------------------------------------|---------|---------|---------|---------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ              | gsm8k   | 0.564   | 0.584   | 500     | accuracy  | {'do_sample': True, 'generation_seed': 42, 'max_new_tokens': 500, 'no_repeat_ngram_size': 0, 'stopping_criteria': [<llmsearch.scripts.stopping_criteria.MultiTokenStoppingCriteria object at 0x7f8f9e357c40>], 'top_k': 10, 'top_p': 0.8} | [metric_file](/Users/praful932/myfiles/code/llmsearch/examples/gsm-8k-best-params-150s-capybara-7b.json)  |
| Praful932/dolphin-2.2.1-mistral-7b-samsum-ft-v1-awq     | samsum  | 0.25543 | 0.25903 | 500     | rouge_2   | {'do_sample': True, 'generation_seed': 42, 'max_new_tokens': 70, 'no_repeat_ngram_size': 0, 'stopping_criteria': [<llmsearch.scripts.stopping_criteria.MultiTokenStoppingCriteria object at 0x7f3b38303610>], 'temperature': 0.1, 'top_k': 50}  | [metric_file](/Users/praful932/myfiles/code/llmsearch/examples/samsum-best-params-500s-tune-capybara-7b.json)  |



## Recommendations
1. **Generative Tasks** : Searching for generation parameters is generally more useful for tasks involving variable-length text outputs rather than tasks with constrained, discrete outputs.
2. **Batch Size** : The batch size can affect performance during evaluation due to the padding tokens' impact on next-token generation. Evaluate the right batch size settings for your specific use case.
3. **Stopping Criteria** : Use a stopping criteria while evaluating models so that model does not endlessly generate tokens until `max_new_tokens` is reached. All of the examples in the repo use a [stopping criteria]().

## Reproducibility
- Running a hyperparameter search works best when results are reproducible (you get similar outputs across different runs).
- There are quantization frameworks such as `exllama` offer very high inference speed while trading against reproducibility. `AWQ` is one of those quantization methods that is fast and also offers reproducibility (This has been used in end2end examples shared above). Note that there may be usecase that may work well even with `exllama` - problems where there is a soft spot for certain generation parameters.
- For the generation to be reproducible there is a `generation_seed` parameter that has been introduced in the `model.generate` method of `transformers` module via monkey patching. This is used while running inference to seed outputs, This can be also treated as a hyperparameter in itself.
- Batch size can affect performance during evaluation, most decoder models use `left` padding, The presence of pad token can affect the next token samples that is generated although very minutely, the effect becomes more pronounced over long sequences. So be sure to evaluate your model at the right batch size.

## Important Considerations
- `llmsearch` modifies certain modules of the transformers library to work with scikit-learn, including:
    - Generation Modules - To support additional generation strategies (`tfs`, `top_a`) and the `generation_seed` parameter for reproducibility.
        - `transformers.GenerationMixin._get_logits_warper` - Older module available at `transformers.GenerationMixin._get_logits_warper_old`
        - `transformers.GenerationConfig.__init__` - Older constructor available via `transformers.GenerationConfig.__init__`
    - Stopping Criteria - Added attribute to avoid cloning issues during searches.
        - `StoppingCriteriaList.__sklearn_clone__`
    - Generation - Added `tfs` & `top_a` support.

## References
- Support for `tfs` & `top_a` reference- https://github.com/oobabooga/text-generation-webui/blob/main/modules/sampler_hijack.py
