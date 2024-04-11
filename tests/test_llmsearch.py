"""
Script to test functionality of llmsearch
Works with /archive/runpod_dev_env_setup.sh

Does not do any kind of exhaustive search

Requires - pip install autoawq==0.2.4 autoawq_kernels==0.0.6
"""

import re
import textwrap
from pathlib import Path

import sys
sys.path.append("/workspace/llmsearch/")

import datasets
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from awq import AutoAWQForCausalLM
from transformers import StoppingCriteriaList, AutoTokenizer

from llmsearch.tuner import Tuner
from llmsearch.utils.mem_utils import gc_cuda
from llmsearch.utils.logging_utils import set_verbosity_debug, set_verbosity_info
from llmsearch.utils.model_downloader import download_model_from_hf
from llmsearch.scripts.stopping_criteria import MultiTokenStoppingCriteria

set_verbosity_debug()

def print_lines(n = 2):
    print("\n" * n)


seed = 42
batch_size = 1
bm_sample_size = 5
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

        print("y_pred - ", y_p_answer)
        print("y_true - ", y_t_answer)

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
callbacks_after_inference = [multi_token_stop_criteria_ob.reset]
print(f"During init - {callbacks_after_inference}")

tuner_ob = Tuner(
    model=model,
    tokenizer=tokenizer,
    dataset=bm_samples,
    device="cuda:0",
    batch_size=batch_size,
    tokenizer_encode_args={"padding": "longest", "add_special_tokens": False},
    tokenizer_decode_args={"spaces_between_special_tokens": False},
    scorer=get_score,
    prompt_template="{X}",
    seed=seed,
    column_mapping={"input_cols": ["X"], "eval_cols": ["answer"]},
    callbacks_after_inference=callbacks_after_inference,
)

# gen_params1 = {
#     "max_new_tokens": 500,
#     "stopping_criteria": stopping_criteria,
#     "generation_seed": 42,
# }

# scores_before, outputs_before = tuner_ob.get_score(gen_params1)

# print_lines(3)
# print("Scores before tuning: ", scores_before)
# print_lines(3)

# ------------------- Grid Search -------------------

hyp_space = {
    "max_new_tokens": [500],
    "stopping_criteria": [stopping_criteria],
    "generation_seed": [42],
    "do_sample": [True],
    'temperature' : [0.9],
    'top_k' : [50],
    'top_p' : [0.95],
}

clf = GridSearchCV(
    estimator=tuner_ob.estimator,
    param_grid=hyp_space,
    scoring=tuner_ob.scorer,
    cv=2,
    n_jobs=None,
    verbose=3,
)

clf.fit(X=tuner_ob.dataset["_X"], y=tuner_ob.dataset["_y"])

# scores_after, outputs_after = tuner_ob.get_score(clf.best_params_)

# print_lines(3)
# print("Scores after tuning: ", scores_after)
# print_lines(3)

# temp_model_dir.unlink()
