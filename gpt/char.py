import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    """
    Emits batches of characters.

    Adapted from "https://github.com/karpathy/minGPT".
    """

    def __init__(self, config, data):
        # get characters from the input data
        # tokens
        chars = sorted(list(set(data)))
        # map characters to integer indices;
        # tokens to token IDs
        # vocabulary
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        
        # map integer indices to characters; 
        # decoding
        self.itos = { i:ch for i,ch in enumerate(chars) }

        self.vocab_size = len(chars)
        self.data_size = len(data)
        self.data = data
        # number of tokens for each sequence
        self.block_size = config.block_size

    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        return self.data_size - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx+self.block_size+1]
        # encode every character to an integer
        encoded = torch.tensor([self.stoi[c] for c in chunk], dtype=torch.long)
        # return the chunk and the shifted version as tensors
        x = encoded[:-1] # contains the input tokens
        y = encoded[1:] # contains the output tokens
        return x, y