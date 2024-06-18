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
#SBATCH --ntasks=1
# Set the number of CPUs per task
#SBATCH --cpus-per-task=16
# Set the number of GPUs
#SBATCH --gpus=2
# Set the amount of memory
#SBATCH --mem=50GB
# Set the time limit (hh:mm:ss)
#SBATCH --time=48:00:00
# Set the output file
#SBATCH --output=transformer_pl_%j.out   


# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate darpa_pytorch

cd /projects/bcxi/nathanj/DARPA_pytorch

# Run the training script
python training_pl.py 
