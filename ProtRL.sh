#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=job_%x_%j.out
#SBATCH --error=job_%x_%j.err
#SBATCH --job-name=ProtRL 
#SBATCH --time=02:00:00


#SBATCH --gres=gpu:7g.80gb:1
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --qos=shorter


set -e
set -u
set -o pipefail

source .env/bin/activate

label="4.2.1.1"
model_directory="AI4PD/ZymCTRL" # put the path to your local model or a Huggingface's repository (to be called with transformer's API)
max_iteration_num=30

echo RL for the enzyme class $label


for i in $(seq 0 $max_iteration_num); # remove to 0 once debugged

do

    echo Starting iteration $i    
    if [ $i != 0 ]; then
    
      echo Train started
      python train.py --iteration_num $i --label $label --model_dir $model_directory --max_iteration_num $max_iteration_num
    
    fi

    echo Sequence generation started
    python seq_gen.py --iteration_num $i --label $label 

    echo dataset generation 
    python dataset_gen.py --iteration_num $i --label $label --model_dir $model_directory 
    
done

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
