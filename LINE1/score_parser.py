import pandas as pd 
import numpy as np

def fasta_yielder(fasta_file): 
    '''
    When provided with the path of a fasta file, yields a tuple with the header and the sequence  
    '''
    with open(fasta_file, 'r') as fasta:
        seq = ""
        identifier = ""
        for line in fasta:
            if line[0] == ">":
                if identifier != "":
                    yield (identifier, seq)
                identifier = line.strip()
                seq = ""
            else:
                seq += line.strip()
        if identifier != "" and seq != "":
            yield (identifier, seq)


seqdict = {
    'prompt': [],
    'sequence': [],
    'advantage': []
}
for identifier, seq in fasta_yielder('L1Seqs_DPO.fasta'): 
    seqdict['prompt'].append('1M')
    seqdict['sequence'].append('1'+seq+'2')
    seqdict['advantage'].append(float(identifier.split('_')[-1]))
    

# make csvs for training, one with fold change as it is, another normalised 
df = pd.DataFrame.from_dict(seqdict)
df.to_csv('training_csv_FC.csv', index=False)

df['advantage'] = df['advantage'] / np.linalg.norm(df['advantage'])
df.to_csv('training_csv_normFC.csv', index=False)