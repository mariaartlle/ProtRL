<img src="https://github.com/user-attachments/assets/6857cfe5-8b43-4a7c-aeea-e1e55eb04c73" width="600"  text-align="center">

# ProtRL: Direct Preference Optimization for Protein Language Models

ProtRL is a Reinforcement Learning (RL) framework for autoregressive protein Language Models (pLMs).
Currently we have implemented the following algorithms: 
- Weighted DPO
- Ranked DPO
- GRPO

This is the repository for the paper [*Guiding Generative Protein Language Models with Reinforcement Learning*](https://arxiv.org/abs/2412.12979). 

### Table of Content
- [About ProtRL](#about-dpo_plm)
- [Installation](#installation)
- [Example](#example)
- [General Usage](#generalusage)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Citation](#citation)

## About ProtRL

This implementation allows you to: 
- train online with a scoring fuction set by you
- train offline with experimental data

Starting from GRPO implementation in [Hugging face](https://huggingface.co/docs/trl/main/en/grpo_trainer), we have implemented a new version to pass custom databases each iteration, and weighted and ranked version of DPO (currently non present in Hugging Face Trainer).

There are two different use cases of this script:
1 -  To train online, each iteration set a standard datasets according HF to have a reward, to then be used. Each iteration generate sequences and save it as: prompt, completition and reward. The reward, will be then used to feedback the model. 
2 - To train on experimental data

### Online training

In case of GRPO, and in case of simple rewards function (leght, aa ratios, hydrophobicity...) you can directly use GRPO HF standard impelentation. For example in case of lenghts:

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
In many cases, the reward can be very complex and for that reason we have implemented a different version of GRPO, where at each iteration, sequences are explicity generated, saved and scored. This is particulary useful in case of we would like to run in arrays in cpu's.
In this case, the reference model must be explicity passed, while the reward function can be set to none. Underhood, this script takes the dataset, and consider the rewards as well. (in the original implementation of GRPO only prompts and completitions where considered, and they are used to align the model to the desired objective. 
For weighted DPO and ranked DPO, the application and the structure is the same, but the loss function slightly change. 

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

For orignal DPO we reccomend HF implementaion. 

This 3 different loss functions were adapted from the firsts described in [Widatalla et al., 2024](https://www.biorxiv.org/content/10.1101/2024.05.20.595026v1.abstract). You can find detailed explanations for each loss function and its changes in formulation in the Methods section of the [paper](https://arxiv.org/abs/2412.12979).

*Note*: Weights and advantages are treated as "the higher, the better." If your scoring function is designed to be minimized, please multiply it by -1.

## Installation

The software needed to run ProtRL can be found in `requeriments.txt`. To set up the environment, execute the following command inside your desired working environment:

```bash
git clone https://github.com/AI4PDLab/ProtRL.git
cd ProtRL
pip install -r requirements.txt
```

## Example 

In the folder `example` a very simple scripr is reported, with the objective to reduce the lenght over the different iterations using a tiny-llama. Given the size of the model this can be run in a 10GB gpu locally. 

To set it up, install the requirements pip install requirements, and run the script from your terminal as bash ProtRL_tiny.sh. It will automatically generate a tiny lama model and plot the results in a graph.  

ProtRL is reported as a very simple script with the objective of decreasing the length over the different iterations to reach a length of 60 amino acids. In the `Experiments` folder, you can find the scripts for experiments that implement more complex scoring functions such as protein folds, functional annotation of enzymes, and experimental data. If you are interested in optimizing for other protein features, you can use `DPO_pLM.py` as a template for your custom RL experiments.

First of all, you will need to set up ZymCTRL or the pLM of your choice. In our case, we downloaded the [HuggingFace's ZymCTRL](https://huggingface.co/AI4PD/ZymCTRL) repository locally or used it directly from the repo, taking advantage of Huggingface's `transformers` API (AI4PD/ZymCTRL). 

With this simple task, we observe that the three modes achieve the desired goal within just a few iterations. While the paired and ranked modes reach the objectives more quickly, they are more prone to catastrophic forgetting compared to the weighted mode. The weighted mode proves to be more stable, particularly in low-data scenarios. It is likely that, with a more complex scoring function and additional data, the ranked and paired algorithms could demonstrate improved performance and behavior.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/e7e89f70-dad0-4731-afbf-37970718e92a" />


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

 


