import json
import argparse
from transformers import AutoTokenizer
import os

#TODO: Implement 10b config

def save_llama_config_json(config_dict, filename):
    with open(filename, "w") as f:
        json.dump(config_dict, f, indent=2)


parser = argparse.ArgumentParser()
parser.add_argument('--model-size', '-s', type=str, nargs='?', default='tiny')
parser.add_argument('--max-pos-embedding', '-p', type=int, nargs='?', default=2048)

# Extract arguments
args = parser.parse_args()
model_size = args.model_size
max_position_embedding = args.max_pos_embedding

# load tokenizer to get values
root_dir = os.path.dirname(os.path.abspath(__file__))
token_dir = os.path.join(root_dir, "models", "tokenizer")
tokenizer = AutoTokenizer.from_pretrained(token_dir)

assert model_size in ['tiny', '1b', '3b', '10b'], "Provide a valid model size: 'tiny' (20m), '1b', '3b' or '10b' "

# 20m paramer model
if model_size == 'tiny':
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 8,
        "vocab_size": tokenizer.vocab_size,
        "max_position_embedding": max_position_embedding,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "tie_word_embeddings": False
    }


# Taken from Noelia colab
elif model_size == '1b':
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": tokenizer.vocab_size,
        "max_position_embedding": max_position_embedding,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "tie_word_embeddings": False
    }

# NOTE: Need check if it actually is for 7b
elif model_size == '7b':
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": tokenizer.vocab_size,
        "max_position_embedding": max_position_embedding,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "tie_word_embeddings": False
    }


config_dir = os.path.join(root_dir, "models", "size_config", model_size)
os.makedirs(config_dir, exist_ok=True)
config_file = os.path.join(config_dir, "llama_config.json")
save_llama_config_json(config, config_file)
