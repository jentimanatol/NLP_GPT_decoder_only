
import torch
from torch.utils.data import Dataset
import random

class CharTokenizer:

    def __init__(self):
        self.stoi={}
        self.itos={}

    def fit(self,text):
        vocab=sorted(set(text))
        self.stoi={ch:i for i,ch in enumerate(vocab)}
        self.itos={i:ch for ch,i in self.stoi.items()}

    def encode(self,text):
        return [self.stoi[c] for c in text]

    def decode(self,ids):
        return ''.join([self.itos[i] for i in ids])

    @property
    def vocab_size(self):
        return len(self.stoi)

class LMDataset(Dataset):

    def __init__(self,tokens,block_size):
        self.tokens=tokens
        self.block_size=block_size

    def __len__(self):
        return len(self.tokens)-self.block_size

    def __getitem__(self,i):
        chunk=self.tokens[i:i+self.block_size+1]
        x=torch.tensor(chunk[:-1])
        y=torch.tensor(chunk[1:])
        return x,y
