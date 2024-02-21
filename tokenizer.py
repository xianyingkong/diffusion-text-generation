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
    def __init__(self, vocab, config_name):
        if vocab == 'bert':
            tokenizer = AutoTokenizer.from_pretrained(config_name)
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            
        elif vocab == 'shakespeare_plays':
            tokenizer = BertTokenizerFast('shakespeare-tokenizer-bert/plays/vocab.txt')
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            
        elif vocab == 'shakespeare_all':
            plays_tokenizer = BertTokenizerFast('shakespeare-tokenizer-bert/plays/vocab.txt')
            sonnets_tokenizer = BertTokenizerFast('shakespeare-tokenizer-bert/sonnets/vocab.txt')
            new_tokens = list(set(sonnets_tokenizer.vocab.keys())-set(plays_tokenizer.vocab.keys()))
            n_new_tokens = plays_tokenizer.add_tokens(new_tokens)
            print(f'### Total of {n_new_tokens} new tokens added')
            self.tokenizer = plays_tokenizer
            self.sep_token_id = plays_tokenizer.sep_token_id
            self.pad_token_id = plays_tokenizer.pad_token_id

        elif vocab == 'combined':
            # look into common english words
            # modern english to old english encoder decoder
#             tokenizer = AutoTokenizer.from_pretrained(config_name)
            tokenizer = BertTokenizerFast('commonsense-tokenizer-bert/vocab.txt')
            ss_tokenizer = BertTokenizerFast('shakespeare-tokenizer-bert/vocab.txt')
            new_tokens = list(set(ss_tokenizer.vocab.keys())-set(tokenizer.vocab.keys()))
            n_new_tokens = tokenizer.add_tokens(new_tokens)
            print(f'### Total of {n_new_tokens} new tokens added')
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