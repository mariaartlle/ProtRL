#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=job_%x_%j.out
#SBATCH --error=job_%x_%j.err

# job name
#SBATCH --job-name=genseqL1FT3

# time limit in seconds
#SBATCH --time=12:00:00

# queue
#SBATCH --qos=normal

# resources
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.40gb:1
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
module load Python/3.10.4-GCCcore-11.3.0

###############
# run command #
###############
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

models_num=( 10 )

for i in "${models_num[@]}"; 
do
    python sequence_generation.py --name FT3 --savedir /users/nferruz/martigues/DPO_pLM/LINE1/RL_FT3/generated_sequences/ --model_dir epoch_"${i}"

done


###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
