from tokenizers import Tokenizer, pre_tokenizers, models
from tokenizers import Tokenizer
import os
from transformers import LlamaTokenizerFast

"Script to train/build a simple LlamaTokenizerFast for a pLM"

# ------ Build a Tokenizer form scratch ------
# Define Tokenizer object
my_tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
# Define how to split input sting
my_tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")

# Initialise the correct tokeinzer trainer
trainer = my_tokenizer.model.get_trainer()

# Add special tokens
spec_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "[UNK]"]
trainer.special_tokens = spec_tokens

# Set vocabulary size

corpus = ['A',
          'B',
          'C',
          'D',
          'E',
          'F',
          'G',
          'H',
          'I',
          'K',
          'L',
          'M',
          'N',
          'O',
          'P',
          'Q',
          'R',
          'S',
          'T',
          'U',
          'V',
          'W',
          'X',
          'Y',
          'Z']

trainer.vocab_size = len(corpus+spec_tokens)

# Build tokenizer from interator
my_tokenizer.train_from_iterator(corpus, trainer=trainer)

# ------- Covert Toknezier obj to FastTokenizer compatible with transformers --

# NOTE: Use specific LlamaTokenizerFast class which has "add_special_token"
# functionality already set-up (and more ?) allow easy training.
my_fast_tokenizer = LlamaTokenizerFast(tokenizer_object=my_tokenizer)
#my_fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=my_tokenizer)

my_fast_tokenizer.add_special_tokens({
    'pad_token': '<|pad|>',
    'bos_token': '<|bos|>',
    'eos_token': '<|eos|>'
})

# Save fasttokenizer directly
root_dir = os.path.dirname(os.path.abspath(__file__))
token_dir = os.path.join(root_dir, "models")
os.makedirs(token_dir, exist_ok=True)
token_dir = os.path.join(token_dir, "tokenizer")
my_fast_tokenizer.save_pretrained(token_dir)
