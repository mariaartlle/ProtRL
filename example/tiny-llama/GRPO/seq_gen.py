import torch
import numpy as np
import os
from tqdm import tqdm
import math
import argparse
import json
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from accelerate.utils import set_seed
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    set_seed(seed)


def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids,model,tokenizer):
    "This function computes perplexities for the generated sequences"
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)


def main(label, model, device,tokenizer):

    # Generating sequences
    input_ids = tokenizer.encode(label,return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids,
        top_k=9, #tbd
        repetition_penalty=1.2,
        max_length=500,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        num_return_sequences=20) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.

    # Check sequence sanity, ensure sequences are not-truncated.
    # The model will truncate sequences longer than the specified max_length (1024 above). We want to avoid those sequences.
    #new_outputs = [ output for output in outputs if output[-1] == 0]
    #if not new_outputs:
    #    print("not enough sequences with short lengths!!")

    # Compute perplexity for every generated sequence in the batch

    # NOTE: IF use different prompts with variable len, need adapting!
    prompt_len = input_ids.shape[-1]
    ppls = [(tokenizer.decode(output[prompt_len:], skip_special_tokens=True), calculatePerplexity(output, model, tokenizer)) for output in outputs]

    # Sort the batch by perplexity, the lower the better
    ppls.sort(key=lambda i:i[1]) # duplicated sequences?

    # Final dictionary with the results
    sequences={}
    sequences[label] = [(x[0], x[1]) for x in ppls]

    return sequences

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int)
    parser.add_argument("--label", type=str)
    args = parser.parse_args()
    iteration_num = args.iteration_num
    ec_label = args.label
    labels = [ec_label.strip()]

    device = torch.device("cuda") # Replace with 'cpu' if you don't have a GPU - but it will be slow
    print('Reading pretrained model and tokenizer')
    
    root_dir = os.path.dirname(os.path.abspath(__file__))

    tokenizer_dir = os.path.join(root_dir, "models", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                              add_eos_token=False, # NEED this for training NOT for generate() else add eos at the end of promt
                                              add_bos_token=True,
                                              use_fast=True)
    
    if iteration_num == 0:
        config_dir = os.path.join(root_dir, "models", "size_config", 'tiny')
        config_file = os.path.join(config_dir, "llama_config.json")

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        config = LlamaConfig(**config_dict)
        model = LlamaForCausalLM(config)

        # Save model on first interation to have a ref model that can be loaded from GRPO Trainer
        ref_model_path = os.path.join(root_dir, "models", "base", "tiny")
        os.makedirs(ref_model_path, exist_ok=True)
        model.cpu().save_pretrained(ref_model_path, from_pt=True)

        # put model to GPU
        model.to(device)
    else:
        model_name = os.path.join(root_dir, f'output_iteration{iteration_num}')
        model = LlamaForCausalLM.from_pretrained(model_name).to(device)

    print('model loaded')

    label = ec_label
    
    
    for label in tqdm(labels):
        all_sequences = []
        seq_len = []
        for i in range(10):
            sequences = main(label, model, device, tokenizer)
            for key, value in sequences.items():
                for index, val in enumerate(value):
                    if len(val[0].replace(" ","")):
                        sequence_info = {
                            'label': label,
                            'batch': i,
                            'index': index,
                            'pepr': float(val[1]),
                            'fasta': f">{label}_{i}_{index}_iteration{iteration_num}\t{val[1]}\n{val[0].replace(" ","")}\n"
                            }
                        all_sequences.append(sequence_info)
                        seq_len.append(len(val[0].replace(" ","")))

        # store seq len statistics
        seq_len = np.array(seq_len)
        mean_len = seq_len.mean()
        std_len = seq_len.std()
        len_stats = np.stack((mean_len, std_len))
        len_stats_dir = os.path.join(root_dir, 'data', 'results')
        os.makedirs(len_stats_dir, exist_ok=True)
        len_stats_file = os.path.join(len_stats_dir, f"len_stats_{label}_iteration_{iteration_num}.npy")
        np.save(len_stats_file, len_stats)

        fasta_content = ''.join(seq['fasta'] for seq in all_sequences)
        seq_dir = os.path.join(root_dir, "data", "inputs")
        os.makedirs(seq_dir, exist_ok=True)
        seq_file = os.path.join(seq_dir, f"seq_gen_{label}_iteration{iteration_num}.fasta")
        fn = open(seq_file, "w")
        fn.write(str(fasta_content))
        fn.close()
