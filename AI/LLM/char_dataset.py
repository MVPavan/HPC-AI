from imports import Path, torch

class CharDataset:
    def __init__ (self, file_path:Path,split_percent:float = 0.9):
        self.file_path = file_path
        self.load_dataset()
        self.prepare_dataset()
        self.train_val_split(split_percent=split_percent)

    def load_dataset(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()
        assert self.data is not None, "Empty input file"
    
    def prepare_dataset(self):
        self.chars = sorted(list(set(self.data)))
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for i,ch in enumerate(self.chars)}
        print(f'Dataset Vocab size - {len(self.chars)}\nVocab: ',''.join(self.chars))
        print('Total Dataset len in chars: ', len(self.data))

    def encode(self, string:str):
        return [self.stoi[ch] for ch in string]

    def decode(self, embed:list[int]):
        return "".join([self.itos[i] for i in embed])
    
    def train_val_split(self,split_percent:float = 0.7):
        n = int(split_percent*len(self.data))
        self.train_data = torch.tensor(self.encode(self.data[:n]), dtype=torch.long)
        self.val_data = torch.tensor(self.encode(self.data[n:]), dtype=torch.long)
    
    def get_batch(self, context_length:int=8, batch:int=4, test:bool=False):
        '''
        Scenario is for next char prediction
        x - Tensor of shape B,T ( T represet time == context length)
        y - Tensor of shape B,T
        such that y contains predictions for x from contexts 1 to T
        example:
            x = [1,2,3,4,5]
            y = [2,3,4,5,6]
            So:
                1 in x -> 2 in y
                1,2 in x -> 3 in y
                1,2,3 in x -> 4 in y
                ...
                1,2,3,4,5 in x -> 6 in y
        '''
        data = self.val_data if test else self.train_data
        # randint high = len(data)+1 -(context+1)#for y
        idx = torch.randint(0,len(data)-context_length, (batch,))
        x = torch.stack([data[i:i+context_length] for i in idx]) # (B,T)
        y = torch.stack([data[i+1:i+1+context_length] for i in idx]) # (B,T)
        return x,y


if __name__ == '__main__':
    tiny_shakes = Path(__file__).parent/'./tiny_shakespeare.txt'
    dataset = CharDataset(file_path=tiny_shakes)
    print('here')