# Setup dev environment
# image - runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04, works with A4000

curl -sSL https://install.python-poetry.org | python3 - && export PATH="~/miniconda3/bin:${PATH}" && conda init bash

# will need to restart terminal
export PATH="/root/.local/bin:${PATH}"
conda create --name llmsearch-env python=3.10 -y && conda activate llmsearch-env && poetry install --extras "pynvml" --with dev && pip install transformers==4.41.1 && pip install torch@https://download.pytorch.org/whl/cu121/torch-2.2.0%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=c441021672ebe2e5afbdb34817aa85e6d32130f94df2da9ad4cb78a9d4b81370

# override transformers & torch instalation

# if using awq
pip install autoawq==0.2.4 autoawq_kernels==0.0.6
# pip install autoawq==0.2.5 autoawq_kernels==0.0.6

# if using gptq
# pip install auto_gptq==0.7.1

# if using exllama
# pip install exllamav2@https://github.com/turboderp/exllamav2/releases/download/v0.0.14/exllamav2-0.0.14+cu121-cp310-cp310-linux_x86_64.whl