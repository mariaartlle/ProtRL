#!/bin/bash -l

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Fail if any command in a pipeline fails

PWD="$(pwd)"
REF_MODEL="/models/base/tiny"
TOKENIZER="models/tokenizer/"
LLAMA_CONFIG="models/size_config/tiny/llama_config.json"
TOKENIZER_PATH="${PWD%/}/${TOKENIZER}"
MODEL_DIRECTORY="${PWD%/}/${REF_MODEL#/}"
CONFIG_PATH="${PWD%/}/${LLAMA_CONFIG#/}"
MAX_ITERATION_NUM=30

DPO_mode="weighted" # choose between paired, ranked and weighted 
label="MDEMKAYVAL"



echo "Create LLaMA3 config file and tokenizer if not there"

if [ -d "$TOKENIZER_PATH" ]; then
    echo "Tokenizer already created"
else
    echo "Creating tokenizer"
    python build_llama_tokenizer.py
fi

if [ -f "$CONFIG_PATH" ]; then
    echo "LLaMA config already created"
else
    echo "Creating LLaMA config"
    python create_llama_config.py -s 'tiny' -p 1024
fi

echo "RL for the enzyme class $label"

for i in $(seq 0 $MAX_ITERATION_NUM); do
    echo "Starting iteration $i"

    if [ $i != 0 ]; then
        echo "Train started"
        python train.py --iteration_num $i --label $label --mode $DPO_mode --model_dir $MODEL_DIRECTORY --max_iteration_num $MAX_ITERATION_NUM
    fi

    echo "Sequence generation started"
    python seq_gen.py --iteration_num $i --label $label

    python plot_len_stats.py
done
