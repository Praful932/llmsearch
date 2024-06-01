"""
Script to test different gen types
Works with /archive/runpod_dev_env_setup.sh

Test Runs Different generation params on the model
Useful to know if monkey patch breaks anything on new versions of transformers
"""

import re
import textwrap
from pathlib import Path

import sys

import torch
import datasets

from tqdm.auto import tqdm
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from transformers import AutoTokenizer, GPT2LMHeadModel

from llmsearch.tuner import Tuner
from llmsearch.utils.mem_utils import gc_cuda
from llmsearch.utils.common_utils import yaml_load
from llmsearch.utils.model_utils import get_device
from llmsearch.utils.model_downloader import download_model_from_hf
from llmsearch.utils.logging_utils import set_verbosity_debug

set_verbosity_debug()

seed = 42
batch_size = 1
model_id = "gpt2-large"
device = get_device()
print(f"Device : {device}")



def load_model_and_tokenizer(model_id, temp_model_dir):
    temp_model_dir.mkdir(exist_ok=True, parents=True)
    output_folder = download_model_from_hf(model_id, save_dir=temp_model_dir, branch="main")

    gc_cuda()

    model = GPT2LMHeadModel.from_pretrained(
        output_folder, device_map={"": device}, local_files_only=True, torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        output_folder, local_files_only=True, legacy=False, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_score(y_true, y_pred):
    scores = []
    for y_t, y_p in zip(y_true, y_pred):
        if y_t['answer'].strip().lower() == y_p.strip().lower():
            scores.append(1.0)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)

# Load Model, Tokenizer, Dataset

temp_model_dir = Path(f"./temp_dir/")
temp_model_dir.mkdir(exist_ok=True, parents=True)

model, tokenizer = load_model_and_tokenizer(model_id, temp_model_dir)
print("Downloaded model")

bm_samples = datasets.Dataset.from_dict(
    {
        "X": ["What is the capital of Germany?", "What is the capital of France?"],
        "answer": ["Berlin", "Paris"],
    }
)

print("Instantiated Tuner")
tuner_ob = Tuner(
    model=model,
    tokenizer=tokenizer,
    dataset=bm_samples,
    device=device,
    batch_size=batch_size,
    tokenizer_encode_args={"padding": "longest", "add_special_tokens": False},
    tokenizer_decode_args={"spaces_between_special_tokens": False},
    scorer=get_score,
    prompt_template="{X}",
    seed=seed,
    column_mapping={"input_cols": ["X"], "eval_cols": ["answer"]},
)

gen_param_list =  yaml_load(Path(__file__).parent / 'test_gen_params.yaml')

print("Starting Generation")

for idx, gen_params in tqdm(enumerate(gen_param_list)):
    print(f"Gen Params: {gen_params}")
    gen_params.pop('stopping_criteria', None)
    score, outputs = tuner_ob.get_score(gen_params)

    print(f"Score: {score}")
    print(f"Outputs: {outputs}")

    print("\n\n")
    print("---" * 15)
    print('\n\n')