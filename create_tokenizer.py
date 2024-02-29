import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast

def create_tokenizer(corpus_path: str, vocab_size: int = 32000, preprocessing=None, save_folder=False):
    """ Creates a tokenizer from given corpus_path and vocab_size.
    Arguments:
        corpus_path (str) - path leading to a file containing all words to be included in vocabulary.
        vocab_size (int) - size of vocabulary for tokenizer
        preprocessing (func) - a function to preprocess/clean data contained in corpus_path.read()
        save_folder (None/str) - if str, saves to specified folder path.
    """
 
    # Create and train tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # Bert will be the base tokenizer
    with open(corpus_path, 'r') as corpus:
        corpus_list = corpus.readlines()
        example = corpus_list[len(corpus_list) // 4] # Pick a line for example
        
        # Preprocess if needed
        if preprocessing is not None:
            corpus_list = preprocessing(corpus.read())

        new_tokenizer = bert_tokenizer.train_new_from_iterator(corpus_list, vocab_size)

    # Display output for user
    print('New tokenizer created')
    # Sanity check for tokenizer
    
    print(f'Tokenizer contains vocab size {new_tokenizer.vocab_size}')
    test_tokenize = new_tokenizer.tokenize(example)
    test_encode = new_tokenizer.encode(example)
    print(f'Example: {example}')
    print(f'Example Tokenized line (length {len(test_tokenize)}): {test_tokenize}')
    print(f'Example Tokenized Index (length {len(test_encode)}): {test_encode}')

    # Save tokenizer
    if isinstance(save_folder, str):
        new_tokenizer.save_pretrained(save_folder) # Used for saving the tokenizer
        print(f'Tokenizer saved at {save_folder}')
        print(f'Tokenizer can be loaded by BertTokenizerFast(vocab_file="{save_folder}")')

    
    return new_tokenizer
        
    
