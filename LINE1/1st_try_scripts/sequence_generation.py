import torch, argparse, os
from transformers import PreTrainedTokenizerFast
from progen.progen2.models.progen.modeling_progen import ProGenForCausalLM
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--savedir", type=str)
parser.add_argument("--model_dir", type=str)
parser.add_argument("--name", type=str)
args = parser.parse_args()

savedir = str(args.savedir)
model_dir = str(args.model_dir)
name = str(args.name)
prompt = '1M'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

## tokenizer
tokenizer = Tokenizer.from_file('/users/nferruz/martigues/scratch/juan_progen2/FT2_redo/tokenizer_progen2.json')
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
fast_tokenizer.eos_token = '<|eos|>'
fast_tokenizer.pad_token = fast_tokenizer.eos_token

model = ProGenForCausalLM.from_pretrained(f'{model_dir}').to(device)
print('Model loaded')
# Assuming 'model' is your model variable
is_on_cuda = next(model.parameters()).is_cuda
print(is_on_cuda)

input_ids = torch.tensor(fast_tokenizer.encode(prompt)).view([1, -1]).to(device)
j = 0
for i in range(125):
 print(f'Generation {i}/125')
 tokens_batch = model.generate(input_ids, do_sample=True, temperature=0.5, max_length=1280, top_p=0.95, num_return_sequences=40, pad_token_id=0)
 as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
 s = tokenizer.decode_batch(as_lists(tokens_batch))
 with open(f'{savedir}DPO_{name}_{model_dir}_{prompt}_{i}.fasta', 'w') as fasta:
  for samp in s:
   ss = samp[1:].strip('2')
   fasta.write(f'>DPO_{name}_{model_dir}_{j}\n{ss}\n')
   j += 1

