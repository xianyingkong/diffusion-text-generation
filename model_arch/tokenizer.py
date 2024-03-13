from transformers import AutoTokenizer, PreTrainedTokenizerFast, BertTokenizerFast
from utils.data import load_data_text
import torch

class myTokenizer():
    """
    Load tokenizer from bert config or defined BPE vocab dict
    """
    ################################################
    ### You can custome your own tokenizer here. ###
    ################################################
    def __init__(self, vocab, vocab_fp):
        print(vocab)
        if vocab == 'bert':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.tokenizer = tokenizer
        elif vocab == 'custom':
            tokenizer = BertTokenizerFast(vocab_fp)
            self.tokenizer = tokenizer
        elif vocab == 'shakespeare':
            tokenizer = BertTokenizerFast('shakespeare-tokenizer-bert/plays/vocab.txt')
            self.tokenizer = tokenizer

        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = len(self.tokenizer)
    
    def encode_token(self, sentences):
        if isinstance(self.tokenizer, dict):
            input_ids = [[0] + [self.tokenizer.get(x, self.tokenizer['[UNK]']) for x in seq.split()] + [1] for seq in sentences]
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            input_ids = self.tokenizer(sentences, add_special_tokens=True)['input_ids']
        else:
            assert False, "invalid type of vocab_dict"
        return input_ids
        
    def decode_token(self, seq):
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = self.tokenizer.decode(seq)
        else:
            assert False, "invalid type of vocab_dict"
        return tokens


def load_model_emb(hidden_dim, tokenizer):
    ### random emb or pre-defined embedding like glove embedding. You can custome your own init here.
    model = torch.nn.Embedding(tokenizer.vocab_size, hidden_dim)
    torch.nn.init.normal_(model.weight)

    return model, tokenizer


def load_tokenizer(vocab, config_name):
    tokenizer = myTokenizer(vocab, config_name)
    return tokenizer

def create_tokenizer(corpus_path: str, vocab_size: int = 32000, save_folder=False):
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
        print(f'Tokenizer can be loaded by BertTokenizerFast(vocab_file="{save_folder}/vocab.txt")')

    
    return new_tokenizer
