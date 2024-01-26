from pathlib import Path

import torch
import lightning as PL
from tokenizers import Tokenizer
from pydantic import BaseModel

class SelfAttentionParams(BaseModel):
    out_dim:int = 512

class CharDataset:
    def __init__ (self, file_path:Path):
        self.file_path = file_path
        self.load_dataset()
        self.prepare_dataset()

    def load_dataset(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()
        assert self.data is not None, "Empty input file"
    
    def prepare_dataset(self):
        self.chars = sorted(list(set(self.data)))
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for i,ch in enumerate(self.chars)}
        print('Dataset Vocab: ',''.join(self.chars))
        print('Total Dataset len in chars: ', len(self.data))

    def encode(self, string:str):
        return [self.stoi[ch] for ch in string]

    def decode(self, embed:list[int]):
        return "".join([self.itos[i] for i in embed])
    
    def train_val_split(self,split_percent:float = 0.7):
        n = int(split_percent*len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]



if __name__ == '__main__':
    tiny_shakes = Path(__file__).parent/'./tiny_shakespeare.txt'
    dataset = CharDataset(file_path=tiny_shakes)
    print('here')