from imports import (
    Path, torch, F, nn, 
    BaseModel, field_validator
)

from char_dataset import CharDataset

class BiGramParams(BaseModel):
    vocab_size:int
    embed_dim:int = 32
    context_length:int = 8
    device:str = 'cuda'
    lr:float = 1e-2
    train_iterations:int = int(1e4)
    val_iterations:int = 200
    val_freq:int = 500
    batch:int

    @field_validator('device')
    @classmethod
    def device_validator(cls, v:str) -> str:    
        if v=='cuda':
            assert torch.cuda.is_available(),"Cuda not available!"
        elif v!='cpu':
            assert False, 'Unknown Device selection'
        return v

class BiGramModel(nn.Module):
    '''
    Bi Gram Model
    '''
    def __init__(self, params:BiGramParams):
        '''
        Vocab size - C
        Embed_dim - D
        '''
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=params.vocab_size, embedding_dim=params.embed_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=params.context_length, embedding_dim=params.embed_dim)
        self.llm_head = nn.Linear(in_features=params.embed_dim, out_features=params.vocab_size)
        self.params = params

    def forward(self, input, target=None):
        '''
        input - (B,T)
        '''
        # Embedding will replace each int along T with embedding of dim D
        B,T = input.shape
        tokens = self.embedding(input) # (B,T,D)
        
        positions = self.pos_embedding(torch.arange(end=T, device=tokens.device)) # T,D
        tokens = tokens+positions # B,T,D + T,D (Broadcasting) -> B,T,D
        
        logits = self.llm_head(tokens) # B,T,C
        if target is None:
            return logits, None

        # cross entropy expects D or channel to be second dim
        B,T,D = logits.shape
        logits = logits.view(B*T,D)
        target = target.view(B*T)
        loss = F.cross_entropy(input=logits,target=target)
        return logits, loss

    def generate(self, input, max_new_chars:int=10):
        '''
        input - (B,T)
        '''
        for _ in range(max_new_chars):
            logits,_ = self.forward(input=input[:,-self.params.context_length:], target=None) #(B,T,C)
            last_char = logits[:,-1,:] #(B,C)
            last_char_probs = F.softmax(last_char, dim=-1) #(B,C)
            # D represent the probablities of a char in vocab size, here D==len(dataset.chars)
            # So we can construct a prob dist based on last char probabilities and draw a sample from it
            sample_index = torch.multinomial(input=last_char_probs, num_samples=1) #(B,1)
            # Multinomial distribution returns the index of the sample it drew
            # Since our stoi representation is same as intiger index, we can use sample index as it is.
            input = torch.cat([input, sample_index], dim=1)
        return input


class Trainer:
    def __init__(self, model: BiGramModel, dataset:CharDataset, params: BiGramParams):
        self.model = model
        self.params = params
        self.model.to(self.params.device)
        self.dataset = dataset
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.params.lr)
    
    def train(self,):
        for iter in range(self.params.train_iterations):
            x,y = self.dataset.get_batch(context_length=self.params.context_length, batch=self.params.batch)
            x, y = x.to(self.params.device), y.to(self.params.device)
            logits, loss = self.model.forward(input=x, target=y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if iter%self.params.val_freq == 0:
                final_losses = self.eval()
                print(f"Iteration: {iter} - Train loss: {final_losses['train']}, Val loss: {final_losses['val']}")
                self.model.train()

    @torch.no_grad()
    def eval(self):
        final_losses = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(size=(self.params.val_iterations,))
            for iter in range(self.params.val_iterations):
                x,y = self.dataset.get_batch(context_length=self.params.context_length, batch=self.params.batch, test=split=='val')
                x, y = x.to(self.params.device), y.to(self.params.device)
                logits, loss = self.model.forward(input=x, target=y)
                losses[iter] = loss.item()
            final_losses[split] = losses.mean()
        return final_losses


if __name__ == '__main__':
    tiny_shakes = Path(__file__).parent/'./tiny_shakespeare.txt'
    dataset = CharDataset(file_path=tiny_shakes)
    params = BiGramParams(
        vocab_size=len(dataset.chars),
        embed_dim=32,
        context_length=8,
        device='cuda',
        batch=32,
        # train_iterations=int(1000)
    )
    model = BiGramModel(params=params)
    bigram_trainer = Trainer(model=model, dataset=dataset, params=params)
    bigram_trainer.train()

    context = torch.zeros((1, 1), dtype=torch.long, device=params.device)
    batch_gen = bigram_trainer.model.generate(input=context, max_new_chars=500)
    for gen in batch_gen.tolist():
        print('----\n',dataset.decode(gen))
