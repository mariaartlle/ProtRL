
import numpy as np
import os
import random
import argparse
import pandas as pd
import statistics
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)

args = parser.parse_args()

def append_to_csv(name, sequence, iteration_num, output_file):
    file_exists = os.path.exists(output_file) and os.stat(output_file).st_size > 0
    with open(output_file, "a", newline="") as csvfile:
        fieldnames = [
            "name",
            "sequence",
            "lenght",
            "iteration_num",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "name" : name,
            "sequence" : sequence,
            "lenght" : len(sequence),
            "iteration_num": iteration_num,
        })

def generate_dataset(iteration_num, ec_label):

    output_file = "logs.csv"
     
    with open(f"seq_gen_{ec_label}_iteration{iteration_num}.fasta", "r") as f:
        rep_seq = f.readlines()

        
    sequences_rep = dict()
     
    for line in rep_seq:
            if ">" in line:
                name = line.split("\t")[0].replace(">", "").replace("\n", "")
                emb_identifier = line.replace(">", "").replace("\n", "")
            else:
                sequence = line.strip()
                
                append_to_csv(name, sequence, iteration_num, output_file)

generate_dataset(args.iteration_num, args.label)