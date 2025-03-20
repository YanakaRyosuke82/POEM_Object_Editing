#!/bin/bash

# Install PyTorch and related packages :  required for SAM2
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Upgrade transformers: required for QWEN2
pip install --upgrade transformers>=4.37.0

# Install additional packages
pip install notebook ipywidgets opencv-python qwen-vl-utils matplotlib ultralytics platformdirs

# install grounding sam
pip3 install autodistill-grounded-sam-2

# quality of life
pip install tqdm lovely-tensors





