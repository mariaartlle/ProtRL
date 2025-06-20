
# ProtRL: Direct Preference Optimization for Protein Language Models
![imagen](https://github.com/user-attachments/assets/b5040d0c-74de-4627-bd2a-3e6344326ef5, height=500px)

ProtRL is a Reinforcement Learning (RL) framework for autoregressive protein Language Models (pLMs).
Currently we have implemented the following algorithms: 
- Weighted DPO
- GRPO

This is the repository for the paper [*Guiding Generative Protein Language Models with Reinforcement Learning*](https://arxiv.org/abs/2412.12979). 

### Table of Content
- [About ProtRL](#about-protrl)
- [Usage](#usage)
- [Installation](#installation)
- [Example](#example)
- [General Usage](#generalusage)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Citation](#citation)

## About ProtRL

ProtRL allows you to:

- [**Train offline**](#offline-training) on pre-existing experimental data.  
- [**Train online**](#online-training) with custom scoring functions in an iterative loop.

Based on the GRPO implementation in [Hugging Face’s TRL library](https://huggingface.co/docs/trl/main/en/grpo_trainer), we have extended the trainer to support:

1. Passing custom datasets at each iteration  
2. Weighted and ranked variants of DPO (not available in the standard Hugging Face trainer)

### Quickstart Example

```python
from src.utils import *
from src.pLM_weigtedDPO import weighted_DPO
from src.pLM_rankedDPO import ranked_DPO 
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(output_dir="ZymCTRL-wDPO", logging_steps=10)

trainer = pLM_wDPOTrainer( #pLM_rDPOTrainer, pLM_GRPOTrainer
    model= "AI4PD/ZymCTRL",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
```
### Usage
#### Offline training
Use ```train_exp.py```, which expects a CSV file with columns:
- prompt: prompt if any (in case of conditional generation)
- sequence: pre-formatted protein sequences
- advantage: numerical weight for each sequence
  
```python 
python train_exp.py --model_dir "AI4PD/ZymCTRL" --csv "training_data.csv"
```


#### Online training
1. Simple GRPO (Hugging Face)
For straightforward rewards (e.g., sequence length, amino-acid ratios), use the standard GRPO trainer:

```python 
GRPO_trainer:
# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("your_dataset")
split = dataset.train_test_split(test_size=0.80, seed=42, shuffle=True)

train_dataset = split['train']
eval_dataset   = split['test']

# Define the reward function, in this case
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

tokenizer = AutoTokenizer.from_pretrained("AI4PD/ZymCTRL")
tokenizer.padding_side = "left"
tokenizer.eos_token_id = 1
tokenizer.pad_token_id = 0

training_args = GRPOConfig(output_dir="ZymCTRL-GRPO", logging_steps=10)

trainer = GRPO_trainer(
    model= "AI4PD/ZymCTRL",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model() 
```
For complex pipelines—where you explicitly generate, save, and externally score sequences each iteration—use our extended trainers. This is ideal for CPU clusters or custom scoring services:

```python
from src.utils import *
from src.pLM_weigtedDPO import weighted_DPO
from src.pLM_rankedDPO import ranked_DPO 
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(output_dir="ZymCTRL-GRPO", logging_steps=10)

trainer = pLM_wDPOTrainer( #pLM_rDPOTrainer, pLM_GRPOTrainer
    model= "AI4PD/ZymCTRL",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
```

For the original DPO algorithm, we recommend the Hugging Face DPO Trainer.

This 3 different loss functions were adapted from the firsts described in [Widatalla et al., 2024](https://www.biorxiv.org/content/10.1101/2024.05.20.595026v1.abstract). You can find detailed explanations for each loss function and its changes in formulation in the Methods section of the [paper](https://arxiv.org/abs/2412.12979).

*Note*: Weights and advantages are treated as "the higher, the better." If your scoring function is designed to be minimized, please multiply it by -1.

## Installation

```bash
git clone https://github.com/AI4PDLab/ProtRL.git
cd ProtRL
pip install -r requirements.txt
```

## Example 
### TinyLLaMA Length Reduction
The example directory includes ```tiny-llama``` directory, which demonstrates decreasing sequence length to 50 amino acids using a TinyLLaMA model that can be run locally on a single GPU. 

Always show details
```bash
cd example/GRPO
bash ProtRL_tiny.sh
```

This generates a TinyLLaMA model, runs RL training, and plots length reduction over iterations.

### Carbonic Anhydrase Fold Adaptation
We also provide a more complex example in example/ZymCTRL-fold, where the fold of carbonic anhydrase is progressively adapted over RL iterations. In this case esm-fold is required and a GPU of 80GB. 

### Experiments

To reproduce the experiments of our paper, you can find all the scripts in the `Experiments` folder. Given the size and computational needs of pLMs, each one of the experiments were executed in one H100 GPU, with differing times of execution. All the parameters and external data used in the experiments can be found in this repo. The `.sh` scripts can be executed from the same folder to conduct each experiment, they have been built to work on a SLURM based cluster, given the need of GPU-intensive computing. To reproduce the results run: 

```bash
bash experiment_name.sh
```
or 
```bash 
sbatch experiment_name.sh
```
Replace `experiment_name` with the desired experiment script path. Each experiment will produce, fold and calculate statistics for each considered feature.

## General Usage
To reinforce your desired feature, you can define and compute a custom reward function within following these steps:

  1. Add Your Custom Functions: Create your own reward function tailored to the feature you want to optimize.
  2. Calculate the Reward: Use your custom function to compute the reward based on your criteria.
  3. Update the DPO weight: Add the computed reward to the data["weights"] column.

Note: Ensure the correct sign of the reward based on your optimization goal: 
  - Use positive values to maximize the scored value.
  - Use negative values to minimize the scored value.
    
In case your are planning to use CLEAN, you will need to clone and set it up as explained in the official [CLEAN repository](https://github.com/tttianhao/CLEAN), and indicate the path in your code. 

Additional notes: 
- lora adapters


## Troubleshooting

Please take a look at the documentation for more details on how to configure and run your experiments.

Feel free to contribute or raise issues if you encounter any problems! We are working to make it more accessible and detailed
## Work in Progress

Change the batch size
## References

- ESM1v: "Language models enable zero-shot prediction of the effects of mutations on protein function" Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, Alexander Rives; doi: https://doi.org/10.1101/2021.07.09.450648. Computed using https://github.com/seanrjohnson/protein_gibbs_sampler/
- ProteinMPNN: "Robust deep learning–based protein sequence design using ProteinMPNN", J. Dauparas et al. Science378,49-56(2022).DOI:10.1126/science.add2187
- CLEAN: "Enzyme function prediction using contrastive learning". Science379,1358-1363(2023). DOI:10.1126/science.adf2465, GitHub: "https://github.com/tttianhao/CLEAN?tab=readme-ov-file"

## Citation 

If you use ProtRL, please cite our [preprint](https://arxiv.org/abs/2412.12979):

```
@misc{stocco2024guidinggenerativeproteinlanguage,
      title={Guiding Generative Protein Language Models with Reinforcement Learning}, 
      author={Filippo Stocco and Maria Artigues-Lleixa and Andrea Hunklinger and Talal Widatalla and Marc Guell and Noelia Ferruz},
      year={2024},
      eprint={2412.12979},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2412.12979}, 
}
```

 


