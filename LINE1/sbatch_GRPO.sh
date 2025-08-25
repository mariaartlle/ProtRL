#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=job_%x_%j.out
#SBATCH --error=job_%x_%j.err

# job name
#SBATCH --job-name=GRPO_L1FT3

# time limit in seconds
#SBATCH --time=12:00:00

# queue
#SBATCH --qos=normal

# resources
#SBATCH --partition=gpu
#SBATCH --gres=gpu:7g.80gb:1
#SBATCH --mem=16GB


##################################
# make bash behave more robustly #
##################################
set -e
set -u
set -o pipefail


###################
# set environment #
###################

source /users/nferruz/martigues/self_training/benchmarking/CLEAN/app/.clean/bin/activate
# source /users/nferruz/martigues/no_backup/ProtRL/LINE1/.protrl/bin/activate
module load Python/3.10.4-GCCcore-11.3.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

###############
# run command #
###############

model_directory="/users/nferruz/martigues/scratch/juan_progen2/FT3_redo/models/checkpoint-1180" # put the path to your local model or a Huggingface's repository (to be called with transformer's API)
training_csv="training_FC.csv"
eval_csv="eval_FC.csv"
all_csv="all_dataset.csv"

python /users/nferruz/martigues/no_backup/ProtRL/train_exp_GRPO.py --model_dir "${model_directory}" --train_csv "${training_csv}" --eval_csv "${eval_csv}" --num_epochs 10 --beta 0.01 --learning_rate 0.00002
# python /users/nferruz/martigues/no_backup/ProtRL/train_exp_GRPO.py --model_dir "${model_directory}" --csv "${all_csv}" --num_epochs 20 --beta 0.01 --learning_rate 0.00002


###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
