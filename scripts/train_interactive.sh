export VENV_PATH=/dtu/blackhole/14/189044/marscho/venvs/m2m
export CACHE_PATH=/dtu/blackhole/14/189044/marscho/cache
export BETTER_EXCEPTIONS=1

# Activate the virtual environment
source $VENV_PATH/bin/activate

# Set all Hugging Face-related cache paths
export HF_HOME=$CACHE_PATH
export TRANSFORMERS_CACHE=$CACHE_PATH
export DATASETS_CACHE=$CACHE_PATH
export TORCH_HOME=$CACHE_PATH
export TORCH_CACHEDIR=$CACHE_PATH
export CUDA_CACHE_PATH=$CACHE_PATH

# Weights and Biases cache and data
export WANDB_CACHE_DIR=$CACHE_PATH
export WANDB_DATA_DIR=$CACHE_PATH

# General Python cache directory for .pyc files
export PYTHONPYCACHEPREFIX=$CACHE_PATH
export TMPDIR=$CACHE_PATH/tmp

export MAIN_PROCESS_PORT=29600

# Marco shapes
if [ "$1" = "marco" ]; then
    export DATASET_ID="/dtu/blackhole/14/189044/marscho/Mask2Mask/_legacy/data/dataset_HF/$2"
    export MODEL_NAME="timbrooks/instruct-pix2pix"

    accelerate launch --multi_gpu --main_process_port $MAIN_PROCESS_PORT train_instruct_pix2pix.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --dataset_name=$DATASET_ID \
        --resolution=512 \
        --train_batch_size=16 \
        --max_train_steps=5000 \
        --learning_rate=7e-05 --lr_warmup_steps=20 \
        --conditioning_dropout_prob=0.05 \
        --report_to="wandb"  \
        --seed=42 \
        --checkpointing_steps 1000 \
        --checkpoints_total_limit 3 \
        --mixed_precision=bf16 \
        --gradient_accumulation_steps=4 \
        --use_ema \
        --output_dir "instruct-pix2pix-model-$2"   
elif [ "$1" = "VAE" ]; then
    export MODEL_NAME="timbrooks/instruct-pix2pix"
    export DATASET_ID="/dtu/blackhole/14/189044/marscho/Mask2Mask/_legacy/data/dataset_HF/exp12"
    accelerate launch --multi_gpu --main_process_port $MAIN_PROCESS_PORT train_instruct_pix2pix_with_vae.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --dataset_name=$DATASET_ID \
        --resolution=512 \
        --train_batch_size=4 \
        --max_train_steps=20000 \
        --learning_rate=7e-05 --lr_warmup_steps=20 \
        --conditioning_dropout_prob=0.05 \
        --report_to="wandb"  \
        --seed=42 \
        --val_image_url_or_path "/dtu/blackhole/14/189044/marscho/Mask2Mask/_legacy/data/dataset_BASE/dataset_1_shape/triangle/base.png" \
        --validation_prompt "resize the object to 2 times its original size" \
        --checkpointing_steps 1000 \
        --checkpoints_total_limit 3 \
        --enable_xformers_memory_efficient_attention \
        --gradient_accumulation_steps=4 \
        --use_ema \
        --output_dir "instruct-pix2pix-VAE" \
        --num_validation_images 1 \
        --validation_epochs 1
fi