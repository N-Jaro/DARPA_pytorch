#!/bin/bash
#SBATCH --job-name=u_transformer_training   
#SBATCH --time=20:00:00
#SBATCH --partition=a100
#SBATCH --account=bbym-hydro
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --output=u_transformer_%j.out   

# Load required modules
module load anaconda3_gpu
pip install --user lightning

# Variables for the script
TRAIN_DATA_DIR="/projects/bbym/shared/data/commonPatchData/"
VAL_DATA_DIR="/projects/bbym/shared/data/commonPatchData/validation"
CHECKPOINT_DIR="checkpoints"
PROJECT_NAME="U_Transformer_Segmentation"
NAME_ID="Utransformer_256_32_w_val_constant_rate"
BATCH_SIZE=16
LEARNING_RATE=1e-3
NUM_EPOCHS=150
PATCH_SIZE=256
OVERLAP=32
NORM_TYPE="basic"
HUE_FACTOR=0.2
VALID_PATCH_RATE=0.75
AUGMENT=True
NUM_WORKERS=12
DYNAMIC_VALID_PATCH_RATE=True

# Run the training script
python -W ignore training_updated.py \
    --train_data_dir $TRAIN_DATA_DIR \
    --val_data_dir $VAL_DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --project_name $PROJECT_NAME \
    --name_id $NAME_ID \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --patch_size $PATCH_SIZE \
    --overlap $OVERLAP \
    --norm_type $NORM_TYPE \
    --hue_factor $HUE_FACTOR \
    --valid_patch_rate $VALID_PATCH_RATE \
    --augment $AUGMENT \
    --num_workers $NUM_WORKERS \
    --dynamic_valid_patch_rate $DYNAMIC_VALID_PATCH_RATE
