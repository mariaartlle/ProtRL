from src.utils import *
from src.pLM_rankedDPO import ranked_DPO
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from trl.trainer.utils import pad
from torch import nn
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
import argparse
import torch
import numpy as np
import random
import pandas as pd
import math


parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--mode", type=str, required=True)
parser.add_argument("--max_iteration_num", type=int, required=True)


args = parser.parse_args()


CONFIG = {
    "beta": 0.01,
    "seed": 42,
    "learning_rate": 2e-6,
    "batch_size": 30,
    "num_epochs": 1,
    "split_percent": 0.2,
    "adam_betas": (0.9, 0.98),
    "epsilon": 1e-8,
    "adam_decay": 0.1,
}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    set_seed(seed)


def reward_len(completions, **kwargs):
    return 0

def format_sequence(sequence, label):
    return f"{label}<sep><start>{sequence}<end><|endoftext|>"

def generate_dataset(iteration_num, label):

    df = pd.read_csv(f"logs.csv")
    df = df[df["iteration_num"] == (iteration_num - 1) ]
    
    rows = []
    for idx, entry in df.iterrows():
        TM_norm_que = float(entry["TM_norm_que"])
        sequence = entry["sequence"]
        algn = float(entry["algn"])
        lenght_rew = math.exp(-((((algn/len(sequence))-1)**2)/(0.5**2)))

        rows.append({
            "prompt": label,
            "completion": format_sequence(sequence, label),
            "reward": float(TM_norm_que + lenght_rew)
        })
    
    return Dataset.from_list(rows)

seed_everything(CONFIG["seed"])

dataset = generate_dataset(args.iteration_num, args.label)
split = dataset.train_test_split(test_size=CONFIG["split_percent"], seed=CONFIG["seed"], shuffle=True)

train_dataset = split['train']
eval_dataset   = split['test'] 

train_dataset.save_to_disk('./dataset/train2')
eval_dataset.save_to_disk('./dataset/val2')

tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
tokenizer.padding_side = "left"
tokenizer.eos_token_id = 1
tokenizer.pad_token_id = 0

training_args = GRPOConfig( output_dir=f"output_iteration{args.iteration_num}", 
                            logging_steps=100,
                            beta = CONFIG["beta"],
                            num_train_epochs = CONFIG["num_epochs"],
                            do_train = True, 
                            do_eval = True, 
                            eval_strategy = "epoch",
                            eval_steps = 500, 
                            save_steps = 500, 
                            save_total_limit = 2,
                            learning_rate = CONFIG["learning_rate"])

if args.iteration_num > 1 : 
    model = f"output_iteration{args.iteration_num-1}"
    checkpoint = checkpoint_load(f"output_iteration{args.iteration_num-1}")

else:
    model = args.model_dir
    checkpoint = None

lr_list = 0.5 * CONFIG["learning_rate"] * (1 + np.cos(np.linspace(0, np.pi, num=args.max_iteration_num)))
#lr_list = np.linspace(CONFIG["learning_rate"], 0.0, num=args.max_iteration_num)

optimizer, model, scheduler  = load_optimizer_scheduler(model, checkpoint, lr_list[args.iteration_num-1].item(), CONFIG)


training_args = GRPOConfig(output_dir=f"output_iteration{args.iteration_num}", 
                           logging_steps=100,
                           beta=CONFIG["beta"],
                           num_train_epochs = CONFIG["num_epochs"],
                           learning_rate = lr_list[args.iteration_num-1].item(),
                           do_train = True, 
                           do_eval = True, 
                           eval_strategy = "epoch",
                           save_strategy = "steps",                      # ← add this line
                           eval_steps = 500, 
                           save_total_limit = 1,
                           save_steps = 5)

trainer = ranked_DPO(
    model= model,
    ref_model = args.model_dir,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class=tokenizer,
    optimizers = (optimizer, scheduler))

trainer.lr_scheduler       = scheduler
trainer.lr_scheduler_state = None

trainer.train()

trainer.save_model()

torch.cuda.empty_cache()


