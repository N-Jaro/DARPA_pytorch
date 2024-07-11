#!/bin/bash
# Set the account name
#SBATCH -A bcxi-tgirails
# Set the job name
#SBATCH --job-name=U_transformer
# Set the partition
#SBATCH --partition=gpu
# Set the number of nodes
#SBATCH --nodes=1
# Set the number of tasks per node
#SBATCH --ntasks-per-node=3
# Set the number of CPUs per task
#SBATCH --cpus-per-task=16
# Set the number of GPUs
#SBATCH --gpus=3
# Set the amount of memory
#SBATCH --mem=100GB
# Set the time limit (hh:mm:ss)
#SBATCH --time=48:00:00
# Set the output file
#SBATCH --output=transformer_pl_scale_%j.out   

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate darpa_pytorch

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd /projects/bcxi/nathanj/DARPA_pytorch

# Run the training script
srun python training_pl_scale.py --name_id "scale_experiment_further" --checkpoint_file '/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/scale_experiment_20240708_112512/best_model.ckpt'

# Train further for 5 more epochs. Continue from this run: https://wandb.ai/9bombs/U_Transformer_Segmentation_scales/runs/inajyz6q?nw=nwuser9bombs
# checkpoint Path: '/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/scale_experiment_20240708_112512/best_model.ckpt'
# project: 'U_Transformer_Segmentation_scales'
# run_id: 'inajyz6q'


############################# New output #############################
# original Utransformer experiment
# checkpoint Path: '/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/utransformer_experiment_pl_20240705_113001/best_model.ckpt'
# project: U_Transformer_Segmentation
# run_id: u3wvt6vj


############################# Old non-sigmoid ouput #############################
# original Utransformer experiment
# checkpoint Path: '/projects/bcxi/nathanj/DARPA_pytorch/checkpoints/utransformer_experiment_pl_20240618_140955/best_model.ckpt'
# project: U_Transformer_Segmentation
# run_id: 32robwvu
