# need to run conda init before this if installed conda for the first time
# conda create --name llmsearch-env python=3.10
poetry install --with dev
conda activate llmsearch-env