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
sys.path.append("/workspace/llmsearch")

import torch
import datasets

from tqdm.auto import tqdm
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from awq import AutoAWQForCausalLM
from transformers import StoppingCriteriaList, AutoTokenizer

from llmsearch.tuner import Tuner
from llmsearch.utils.mem_utils import gc_cuda
from llmsearch.utils.common_utils import yaml_load
from llmsearch.utils.model_downloader import download_model_from_hf
from llmsearch.scripts.stopping_criteria import MultiTokenStoppingCriteria

seed = 42
batch_size = 1
bm_sample_size = 2
model_id = "TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ"
device = "cuda:0"

def load_model_and_tokenizer(model_id, temp_model_dir):
    temp_model_dir.mkdir(exist_ok=True, parents=True)
    output_folder = download_model_from_hf(model_id, save_dir=temp_model_dir, branch="main")

    gc_cuda()

    model = AutoAWQForCausalLM.from_quantized(
        quant_path=output_folder, fuse_layers=True, device_map={"": device}, local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        output_folder, local_files_only=True, legacy=False, use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"

    return model, tokenizer

def load_dataset():

    def preprocess_dataset(
        dataset, tokenizer, pt, pt_cols, system_prompt, add_generation_prompt=True
    ):

        def wrapper(sample):
            """Takes in a sample, formats it using prompt template, applies chat template and returns the formatted string"""
            messages = (
                []
                if system_prompt is None
                else [{"role": "system", "content": system_prompt}]
            )
            formatted_pt = pt.format(**{pt_col: sample[pt_col] for pt_col in pt_cols})
            messages.append(
                {
                    "role": "user",
                    "content": formatted_pt,
                }
            )
            formatted_pt_with_ct = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            return formatted_pt_with_ct

        def actual_input(sample):
            """Takes in a sample, formats it using prompt template, applies chat template and returns the formatted string"""
            return sample[pt_cols[0]]

        pt_dataset = dataset.map(
            lambda sample: {
                "X": wrapper(sample),
                "actual input": actual_input(sample),
            }
        )

        return pt_dataset


    # 2-shot prompt template - https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml
    pt = textwrap.dedent(
    """\
    Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
    A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

    Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
    A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

    Q: {question}"""
    )
    pt_cols = ["question"]
    system_prompt = "Solve the following math problems, end with The answer is"
    gsm8k_dataset = datasets.load_dataset("gsm8k", "main")


    processed_dataset = preprocess_dataset(
        gsm8k_dataset["train"],
        tokenizer,
        pt=pt,
        pt_cols=pt_cols,
        system_prompt=system_prompt,
        add_generation_prompt=True,
    )
    bm_samples = processed_dataset.shuffle(seed=seed).select(range(bm_sample_size))
    return bm_samples

def get_score(y_true, y_pred):
    def extract_answer_from_out(s):
        pattern = re.compile(r"The answer is (\d+(?:\.\d+)?)")
        match = pattern.search(s)
        if match:
            return match.group(1).strip()
        else:
            return None

    scores = []

    for y_t, y_p in zip(y_true, y_pred):
        y_t_answer = y_t["answer"].split("####")[-1].strip()
        y_p_answer = extract_answer_from_out(y_p)

        if y_t_answer == y_p_answer:
            scores.append(1)
        else:
            scores.append(0)
    return sum(scores) / len(scores)

# Load Model, Tokenizer, Dataset

temp_model_dir = Path(f"./temp_dir/")
temp_model_dir.mkdir(exist_ok=True, parents=True)

model, tokenizer = load_model_and_tokenizer(model_id, temp_model_dir)

bm_samples = load_dataset()

multi_token_stop_criteria_ob = MultiTokenStoppingCriteria(sequence_ids=[32000])
stopping_criteria = StoppingCriteriaList([multi_token_stop_criteria_ob])

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
    callbacks_after_inference=[multi_token_stop_criteria_ob.reset],
)

gen_param_list =  yaml_load(Path(__file__).parent / 'test_gen_params.yaml')

for idx, gen_params in tqdm(enumerate(gen_param_list)):
    print(f"Gen Params: {gen_params}")
    if 'stopping_criteria' in gen_params:
        gen_params['stopping_criteria'] = stopping_criteria
    score, outputs = tuner_ob.get_score(gen_params)

    print(f"Score: {score}")
    print(f"Outputs: {outputs}")

    print("\n\n")
    print("---" * 15)
    print('\n\n')