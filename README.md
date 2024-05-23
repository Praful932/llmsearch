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
- [llama-3-8b Example]() - A Quickstart Notebook shows basic functionality of `llmsearch`

### End-to-End Model Examples
1. [GSM8K Example](https://github.com/Praful932/llmsearch/blob/main/examples/gsm8k_example.ipynb) - Shows a GridSearch ran on the [GSM8K](https://huggingface.co/datasets/gsm8k) Dataset using the `TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ` model.
2. [Samsum Example](https://github.com/Praful932/llmsearch/blob/main/examples/samsum_example.ipynb) - Shows a GridSearch ran on the samsum Dataset
using a finetuned(on samsum dataset) version of `cognitivecomputations/dolphin-2.2.1-mistral-7b`.

## Benchmarks


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

## References
- Tranformers Monkey Patch reference - https://github.com/oobabooga/text-generation-webui/blob/main/modules/sampler_hijack.py
