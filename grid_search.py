# Autoreload
import sys
import beepy
from typing import List

import numpy as np
from IPython.display import Audio, display


import nltk
import torch
import numpy as np
import datasets
import pandas as pd
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, T5ForConditionalGeneration, AutoModelForSeq2SeqLM


import llmsearch


device = "cpu"

if torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

from llmsearch.utils.mem_utils import gc_cuda

print(f"Device - {device}")

def beep(duration = 1, frequency=440, rhythm=1):
    sample_rate = 44100  # Standard audio sample rate
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    audio_data = np.sin(2*np.pi*frequency*t)  # Generate a sine wave
    audio_data *= np.where(np.arange(len(audio_data)) % rhythm == 0, 1, 0)  # Apply rhythm
    display(Audio(audio_data, rate=sample_rate, autoplay=True))

dataset = datasets.load_dataset("samsum")

sample_size = 100
samples_to_tune_on = datasets.Dataset.from_dict(dataset["train"][:sample_size])
samples_to_tune_on = samples_to_tune_on.rename_columns(column_mapping = {'dialogue' : 'X', 'summary' : "y"})


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast = False)
model =  AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

import langchain

X = samples_to_tune_on[0]['X']

pt = langchain.PromptTemplate.from_template("Conversation: {X}\nSummary:")

import evaluate

rouge_metric = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]


    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def get_rouge_score(y_true: List, y_pred: List):
    preds, gts = postprocess_text(preds=y_pred, labels=y_true)

    result = rouge_metric.compute(predictions=preds, references=gts, use_stemmer=True, use_aggregator=False)
    return np.mean(result['rouge2'])


from llmsearch.tuner import Tuner
from llmsearch.utils.mem_utils import get_total_available_ram, get_gpu_information
from llmsearch.utils.logging_utils import set_verbosity_info, set_verbosity_debug
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

seed = 42

set_verbosity_info()

tuner_ob = Tuner(model = model,tokenizer = tokenizer,dataset = samples_to_tune_on,
                 device = device, batch_size = 512,
                 tokenizer_encoding_kwargs={'padding': True, 'truncation': True, 'max_length': 512},
                 tokenizer_decoding_kwargs = {'skip_special_tokens' : True,  'spaces_between_special_tokens' : False},
                 scorer = get_rouge_score, prompt_template = pt, is_encoder_decoder = True, seed = seed, column_mapping = {"text_column_name": "X", "label_column_name": "y"})

# Earlier
from llmsearch.utils.model_utils import seed_everything
"""
parameters and how they affect do_sample == False
1. temperature - output does not change - greedy decoding
2. top_k - output does not change - greedy decoding
3. repetition_penalty - output changes
4. no_repeat_ngram_size - output changes
"""

# seed_everything(seed)

initial_generation_params1 = {
    'max_new_tokens' : 120,
#     'temperature' : 0.7,
#     'do_sample' : True,
#     'generation_seed' : 42,
#     'tfs' : 0.95,
#     'top_a' : 0.3,

#     "epsilon_cutoff": 1.49,
#     "eta_cutoff": 10.42,
#     "repetition_penalty": 1.17,
#     "temperature": 1.31,
#     "top_a": 0.52,
#     "top_k": 49,
#     "top_p": 0.14,
#     "do_sample": True,
#     "generation_seed": 42,
}
score, outputs1 = tuner_ob.get_score(initial_generation_params1)

print(f"Score before tuning - {score}\n")

from llmsearch.utils.gen_utils import get_sample_hyp_space

sample_hyp_spaces = get_sample_hyp_space(seed = 42, max_new_tokens = 120)

hyp_param_grid = sample_hyp_spaces[0]


scorer = make_scorer(score_func=get_rouge_score, greater_is_better=True)

clf = GridSearchCV(
    estimator = tuner_ob.estimator,
    param_grid=hyp_param_grid,
    scoring = scorer,
    cv = 5,
    n_jobs = None,
    verbose=3,
)

print("\n--------- Starting tuning ---------\n")

clf.fit(X=tuner_ob.dataset["X"], y=tuner_ob.dataset["y"])

print("\n--------- Best Params ---------\n")
print(clf.best_params_)

score2, outputs2 = tuner_ob.get_score(clf.best_params_)

print("\n--------- Score after tuning ---------\n")
print(score2)

print("\n--------- Output Diff ---------\n")
for gt, out1, out2 in zip(samples_to_tune_on['y'], outputs1, outputs2):
    if out1 != out2:
        print(f"Ground Truth - {gt}")
        print(f"out1 - {out1}")
        print(f"out2 - {out2}\n\n")

beepy.beep(sound = 1)