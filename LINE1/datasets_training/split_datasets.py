#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

def check_mmseqs_installed():
    """Checks if MMseqs2 is installed and available in the system's PATH."""
    if shutil.which("mmseqs") is None:
        print("Error: MMseqs2 is not installed or not in your PATH.", file=sys.stderr)
        print("Please install it from https://github.com/soedinglab/MMseqs2.", file=sys.stderr)
        sys.exit(1)

def run_mmseqs(fasta_file, work_dir):
    """
    Runs MMseqs2 for an all-vs-all sequence comparison.

    Args:
        fasta_file (str): Path to the input FASTA file.
        work_dir (str): Directory to store MMseqs2 intermediate files.

    Returns:
        str: Path to the TSV results file.
    """
    print("--- Running MMseqs2 ---")
    db_path = os.path.join(work_dir, "proteinDB")
    aln_path = os.path.join(work_dir, "proteinAln")
    results_path = os.path.join(work_dir, "results.tsv")
    
    # Create database
    print(f"1. Creating database from {fasta_file}...")
    subprocess.run(
        ["mmseqs", "createdb", fasta_file, db_path],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # Run all-vs-all search
    print("2. Running all-vs-all search...")
    subprocess.run(
        ["mmseqs", "search", db_path, db_path, aln_path, work_dir],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # Create results TSV
    print("3. Generating results table...")
    subprocess.run(
        ["mmseqs", "createtsv", db_path, db_path, aln_path, results_path, "--full-header"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    print("--- MMseqs2 run complete ---")
    return results_path

def analyze_and_stratified_split(results_tsv, test_split_percentage=20, num_bins=5):
    """
    Analyzes pairwise identities and performs a stratified split to create
    two sets with equivalent diversity distributions based on the desired percentage.

    Args:
        results_tsv (str): Path to the MMseqs2 results TSV file.
        test_split_percentage (int): Percentage of sequences for the smaller set.
        num_bins (int): Number of diversity strata to create for sampling.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The full pairwise identity data.
            - set: A set of IDs for the larger group (e.g., training set).
            - set: A set of IDs for the smaller group (e.g., test set).
    """
    print("--- Analyzing and performing stratified split by diversity ---")
    col_names = [
        "query", "target", "pident", "alnlen", "mismatch",
        "gapopen", "qstart", "qend", "tstart", "tend", "evalue", "bitscore",
        "qheader", "theader"
    ]
    
    df = pd.read_csv(results_tsv, sep='\t', header=None, names=col_names)
    df_no_self = df[df['query'] != df['target']].copy()
    
    # 1. Calculate the average identity for each protein
    print("1. Calculating average identity for each protein...")
    avg_identities = df_no_self.groupby('query')['pident'].mean()
    
    # 2. Bin proteins into diversity strata
    print(f"2. Binning proteins into {num_bins} diversity strata...")
    try:
        bins = pd.qcut(avg_identities, q=num_bins, labels=False, duplicates='drop')
    except ValueError:
        print("Warning: Could not create the desired number of bins due to non-unique identity values. Using fewer bins.")
        bins = pd.qcut(avg_identities, q=num_bins, labels=False, duplicates='raise')


    avg_identities_df = pd.DataFrame({'avg_identity': avg_identities, 'bin': bins})
    
    # 3. Perform stratified sampling from each bin
    print(f"3. Performing stratified sampling with a {100-test_split_percentage}/{test_split_percentage} split...")
    train_set_ids = set()
    test_set_ids = set()
    
    for bin_id in avg_identities_df['bin'].unique():
        bin_proteins = avg_identities_df[avg_identities_df['bin'] == bin_id]
        
        # Use pandas sample method for random sampling
        test_sample = bin_proteins.sample(frac=test_split_percentage / 100.0, random_state=42)
        train_sample_ids = bin_proteins.drop(test_sample.index).index
        
        test_set_ids.update(test_sample.index)
        train_set_ids.update(train_sample_ids)

    print(f"4. Created training set with {len(train_set_ids)} sequences.")
    print(f"5. Created test set with {len(test_set_ids)} sequences.")
    
    return df_no_self, train_set_ids, test_set_ids

def split_fasta_file(original_fasta, train_set_ids, test_set_ids, output_prefix, percentage):
    """
    Splits the original FASTA file into two stratified sets.

    Args:
        original_fasta (str): Path to the original FASTA file.
        train_set_ids (set): Set of IDs for the training group.
        test_set_ids (set): Set of IDs for the test group.
        output_prefix (str): Prefix for the output files.
        percentage (int): The percentage used for the test set split.
    """
    print("--- Splitting FASTA file into stratified sets ---")
    train_percentage = 100 - percentage
    train_fasta_path = f"{output_prefix}_train_{train_percentage}.fasta"
    test_fasta_path = f"{output_prefix}_test_{percentage}.fasta"
    
    train_records = []
    test_records = []
    
    for record in SeqIO.parse(original_fasta, "fasta"):
        if record.id in train_set_ids:
            train_records.append(record)
        elif record.id in test_set_ids:
            test_records.append(record)
            
    SeqIO.write(train_records, train_fasta_path, "fasta")
    SeqIO.write(test_records, test_fasta_path, "fasta")
    
    print(f"Training set saved to: {train_fasta_path}")
    print(f"Test set saved to: {test_fasta_path}")

def plot_identity_distribution(df, train_set_ids, test_set_ids, output_prefix, percentage):
    """
    Plots the distribution of sequence identities for both stratified sets.

    Args:
        df (pd.DataFrame): DataFrame with pairwise identities.
        train_set_ids (set): Set of IDs for the training group.
        test_set_ids (set): Set of IDs for the test group.
        output_prefix (str): Prefix for the output plot file.
        percentage (int): The percentage used for the test set split.
    """
    print("--- Generating identity distribution plot for stratified sets ---")
    
    identities_train = df[
        (df['query'].isin(train_set_ids)) & (df['target'].isin(train_set_ids))
    ]['pident']
    
    identities_test = df[
        (df['query'].isin(test_set_ids)) & (df['target'].isin(test_set_ids))
    ]['pident']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    train_percentage = 100 - percentage
    sns.kdeplot(identities_train, ax=ax, fill=True, 
                label=f'Train Set ({train_percentage}%, {len(train_set_ids)} seqs)', color='darkcyan')
    sns.kdeplot(identities_test, ax=ax, fill=True, 
                label=f'Test Set ({percentage}%, {len(test_set_ids)} seqs)', color='goldenrod', alpha=0.7)

    ax.set_title('Distribution of Pairwise Sequence Identity in Stratified Sets', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sequence Identity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    
    plot_path = f"{output_prefix}_stratified_identity_distribution.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    plt.close()

def main():
    """Main function to orchestrate the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze protein sequence similarity using MMseqs2, "
                    "perform a stratified split into two sets with equivalent diversity, "
                    "and plot their identity distributions."
    )
    parser.add_argument("fasta_file", type=str, help="Path to the input FASTA file.")
    parser.add_argument(
        "-p", "--percentage", type=int, default=20,
        help="Percentage of sequences to allocate to the test set (default: 20)."
    )
    parser.add_argument(
        "-o", "--output_prefix", type=str,
        help="Prefix for output files (default: derived from input file name)."
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.fasta_file):
        print(f"Error: Input file not found at '{args.fasta_file}'", file=sys.stderr)
        sys.exit(1)

    check_mmseqs_installed()

    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = os.path.splitext(os.path.basename(args.fasta_file))[0]
    
    work_dir = f"{output_prefix}_mmseqs_work"
    os.makedirs(work_dir, exist_ok=True)

    try:
        results_tsv = run_mmseqs(args.fasta_file, work_dir)
        df, train_ids, test_ids = analyze_and_stratified_split(results_tsv, args.percentage)
        split_fasta_file(args.fasta_file, train_ids, test_ids, output_prefix, args.percentage)
        plot_identity_distribution(df, train_ids, test_ids, output_prefix, args.percentage)
        print("\nAnalysis complete!")

    except subprocess.CalledProcessError as e:
        print("An error occurred while running MMseqs2.", file=sys.stderr)
        print(f"Command: {' '.join(e.cmd)}", file=sys.stderr)
        print(f"Stderr: {e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        print(f"Cleaning up temporary directory: {work_dir}")
        shutil.rmtree(work_dir)

if __name__ == "__main__":
    main()
