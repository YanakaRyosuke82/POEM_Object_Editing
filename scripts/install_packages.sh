#!/bin/bash

# Install PyTorch and related packages :  required for SAM2
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Upgrade transformers: required for QWEN2
pip install --upgrade transformers>=4.37.0

# Install additional packages
pip install notebook ipywidgets platformdirs opencv-python qwen-vl-utils matplotlib

git clone https://github.com/facebookresearch/sam2.git

cd sam2

pip install -e ".[notebooks]"



#### IT FAILED
# pip install 'git+https://github.com/facebookresearch/sam2.git'

# wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt