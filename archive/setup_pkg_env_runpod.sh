export PATH="~/miniconda3/bin:${PATH}"
export PATH="/root/.local/bin:${PATH}"
conda init bash

conda create --name llmsearch-env python=3.10
conda activate llmsearch-env

poetry install --extras "pynvml" --with dev --no-root
pip install https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl

pip install autoawq@https://github.com/casper-hansen/AutoAWQ/releases/download/v0.2.0/autoawq-0.2.0+cu118-cp310-cp310-linux_x86_64.whl