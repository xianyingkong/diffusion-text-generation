import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast


tokenizer_save_path = 'shakespeare-tokenizer-bert'
data_path = os.path.join('data', 'shakespeare.csv')
vocab_size = 32000 # Change if desired

def create_tokenizer(corpus_path: str, vocab_size: int, preprocessing=None):
    """ Creates a tokenizer from given corpus_path and vocab_size.
    Arguments:
        corpus_path (str) - path leading to a file containing all words to be included in vocabulary.
        vocab_size (int) - size of vocabulary for tokenizer
        preprocessing (func) - a function to preprocess/clean data contained in corpus_path.
    """
    with open(corpus_path
# Open file with iteration
with open(data_path, 'r') as corpus:
    new_tokenizer = bert_tokenizer.train_new_from_iterator(corpus.readlines(), vocab_size)



# Creates new tokenizer with our vocabulary set
new_tokenizer = bert_tokenizer.train_new_from_iterator(lines_iter, 32000)

# Just for sanity check
print(f'Tokenizer contains vocab size {new_tokenizer.vocab_size}')
test_tokenize = new_tokenizer.tokenize(data.PlayerLine.iloc[0])
test_encode = new_tokenizer.encode(data.PlayerLine.iloc[0])
print(f'Example Tokenized line (length {len(test_tokenize)}): {test_tokenize}')
print(f'Example Tokenized Index (length {len(test_encode)}): {test_encode}')

#with open('shakespeare_lines.raw', 'r') as shakespeare_lines:
#    new_tokenizer = bert_tokenizer.train_new_from_iterator(shakespeare_lines, 32000)

# Save
new_tokenizer.save_pretrained("shakespeare-tokenizer-bert")

vocab_path = os.path.join(tokenizer_save_path, 'vocab.txt')
print(f'Tokenizer can be loaded by BertTokenizerFast(vocab_file="{vocab_path}")')