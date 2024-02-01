from imports import (
    Path,math,  torch, F, nn, summary,
    BaseModel, field_validator, Optional
)

from char_dataset import CharDataset

'''
GPT2 Code
Batch - B
Sequence/Context - T
Embed Dim - D
single head k,q,v out dim - head_dim - H
n_heads - N
'''


class AttParams(BaseModel):
    vocab_size:int
    embed_dim:int = 64
    n_heads:int = 4
    mlp_hidden_dim:Optional[int] = None
    n_blocks:int = 4
    batch:int = 16
    context_length:int = 32
    device:str = 'cuda'
    lr:float = 1e-3
    dropout:float = 0.0
    bias:bool = False
    train_iterations:int = int(1e4)
    val_iterations:int = 200
    val_freq:int = 500
    _head_dim:int = 0 # private field

    @field_validator('device')
    @classmethod
    def device_validator(cls, v:str) -> str:    
        if v=='cuda':
            assert torch.cuda.is_available(),"Cuda not available!"
        elif v!='cpu':
            assert False, 'Unknown Device selection'
        return v
    
    def model_post_init(self, __context) -> None:
        # When n_heads concatenated use linear layer to project to -> embed_dim
        self._head_dim = self.embed_dim//self.n_heads 
        assert self.embed_dim % self.n_heads == 0 , 'Embed dim should be proper multiple of num of heads'


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class AttentionHead(nn.Module):
    '''
    '''
    def __init__(self, params:AttParams):
        super().__init__()
        '''
        Combining k,q,v of all heads to one kqv
        in single head each k,q,v has D,H matrix 
        such that concat over all heads gives k,qv each of D,D => H = D/N
        now to combine all k,q,v of all heads to one single kqv in two steps:
            1. k,q,v of single head to one D,H -> D,3*H 
            2. Combining multiple heads D,3*H -> D,3*H*N = D,3*D
        while usage:
            1. Get K,Q,V of all heads D,3*D -> 3(D,D)
            2. For attention view k,q,v as individual heads with heads as seperate dim
                3(D,D) -> 3(D,(N,H))
        '''
        self.w_kqv = nn.Linear(in_features=params.embed_dim, out_features=3*params.embed_dim, bias=params.bias)
        self.proj = nn.Linear(params.embed_dim, params.embed_dim, bias=params.bias)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 1,1 is added to accomodate Batch and Head dimns
            self.register_buffer('tril', torch.tril(torch.ones(1,1,params.context_length, params.context_length)))

        self.attn_dropout = nn.Dropout(params.dropout)
        self.resid_dropout = nn.Dropout(params.dropout)
        self.params = params

    def forward(self, x:torch.Tensor, causal=False):
        # X - B,T,D
        # Out - B,T,H (H = D//N)
        B,T,D = x.shape

        qkv = self.w_kqv(x) # B,T,3*D
        q,k,v = qkv.split(D, dim=2) # 3(B,T,D)
        q,k,v = [m.view(B,T,params.n_heads,params._head_dim).transpose(1,2) for m in (q,k,v)] # 3(B,N,T,H)

        if self.flash:
            out = F.scaled_dot_product_attention(
                query=q,key=k,value=v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=causal
            ) # (B,N,T,H)
        else:
            attention = q@k.transpose(-2,-1)*(1.0/math.sqrt(k.size(-1))) # q(B,N,T,H)@k.T(B,N,H,T)/sqrt(H) -> B,N,T,T
            if causal:
                attention = attention.masked_fill(self.tril[:,:,:T, :T] == 0, float('-inf'))
            # B,N,T,T matrix, convert rows to prob such that attention@v will have attention weighted v
            attention = F.softmax(attention, dim=-1)
            attention = self.attn_dropout(attention)
            out = attention@v # (B,N,T,T)@(B,N,T,H) -> B,N,T,H
        
        out = out.transpose(1,2).view(B,T,D) # B,T,N*H=D
        return out # B,T,D

class MultiheadAttention(nn.Module):
    '''
    Collection of multiple Attention blocks
    '''
    def __init__(self, params: AttParams):
        super().__init__()
        self.atthead_list = nn.ModuleList([AttentionHead(params) for _ in range(params.n_heads)])
        self.project = nn.Linear(in_features=params._head_dim*params.n_heads, out_features=params.embed_dim)
        self.dropout = nn.Dropout(params.dropout)
    
    def forward(self, x):
        # X - B,T,D
        out = torch.cat([atthead(x) for atthead in self.atthead_list], dim=-1) # B,T,D
        out = self.project(out)  # B,T,D
        out = self.dropout(out)
        return out  # B,T,D
    

class FeedForward(nn.Module):
    '''
    Multi layer perceptron
    '''
    def __init__(self, params):
        super().__init__()
        factor = 4 if params.mlp_hidden_dim is None else params.mlp_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features=params.embed_dim, out_features=factor*params.embed_dim),
            nn.ReLU(),
            nn.Linear(in_features=factor*params.embed_dim, out_features=params.embed_dim),
            nn.Dropout(params.dropout)
        )
    
    def forward(self, x):
        return self.mlp(x) # B,T,D

class SelfAttentionBlock(nn.Module):
    '''
    Collection of LayerNorm, Multihead attention, Residual, LayerNorm, FeedForward, Residual
    '''
    def __init__(self, params):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=params.embed_dim)
        self.mha = MultiheadAttention(params=params)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=params.embed_dim)
        self.ffn = FeedForward(params=params)

    def forward(self,x):
        out = self.layer_norm_1(x)  # B,T,D
        out = x + self.mha(x)  # B,T,D
        out = self.layer_norm_2(out)  # B,T,D
        out = x + self.ffn(out)  # B,T,D
        return out  # B,T,D

class LLMHead(nn.Module):
    '''
    Layer norm and final liner prejection to vocab size
    '''
    def __init__(self, params):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=params.embed_dim)
        self.project = nn.Linear(in_features=params.embed_dim, out_features=params.vocab_size)
    
    def forward(self, x):
        return self.project(self.layer_norm(x)) # B,T,V

class GPT(nn.Module):
    '''
    Transformer Model
    '''
    def __init__(self, params:AttParams):
        '''
        Vocab size - V
        Embed_dim - D
        Sequence/Context - T
        '''
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=params.vocab_size, embedding_dim=params.embed_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=params.context_length, embedding_dim=params.embed_dim)
        self.attention_blocks = nn.Sequential(*[SelfAttentionBlock(params=params) for _ in range(params.n_blocks)])
        self.llm_head = LLMHead(params=params)
        self.params = params

    def forward(self, input, target=None):
        '''
        input - (B,T)
        '''
        # Embedding will replace each int along T with embedding of dim D
        B,T = input.shape
        tokens = self.token_embedding(input) # (B,T,D)
        positions = self.pos_embedding(torch.arange(end=T, device=tokens.device)) # T,D
        x = tokens+positions # B,T,D + T,D (Broadcasting) -> B,T,D
        x = self.attention_blocks(x) # B,T,D
        logits = self.llm_head(x) # B,T,V
        if target is None:
            return logits, None

        # cross entropy expects V or channel to be second dim
        B,T,V = logits.shape
        logits = logits.view(B*T,V)
        target = target.view(B*T)
        loss = F.cross_entropy(input=logits,target=target)
        return logits, loss

    def generate(self, input, max_new_chars:int=10):
        '''
        input - (B,T)
        '''
        for _ in range(max_new_chars):
            logits,_ = self.forward(input=input[:,-self.params.context_length:], target=None) #(B,T,V)
            last_char = logits[:,-1,:] #(B,V)
            last_char_probs = F.softmax(last_char, dim=-1) #(B,V)
            # D represent the probablities of a char in vocab size, here D==len(dataset.chars)
            # So we can construct a prob dist based on last char probabilities and draw a sample from it
            sample_index = torch.multinomial(input=last_char_probs, num_samples=1) #(B,1)
            # Multinomial distribution returns the index of the sample it drew
            # Since our stoi representation is same as intiger index, we can use sample index as it is.
            input = torch.cat([input, sample_index], dim=1)
        return input


class Trainer:
    def __init__(self, model: GPT, dataset:CharDataset, params: AttParams):
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
    params = AttParams(vocab_size=len(dataset.chars))
    model = GPT(params=params)
    x,y = dataset.get_batch(context_length=params.context_length, batch=params.batch)
    model_summary = summary(model=model,input_data=(x,y))
    trainer = Trainer(model=model, dataset=dataset, params=params)
    trainer.train()

    context = torch.zeros((1, 1), dtype=torch.long, device=params.device)
    batch_gen = trainer.model.generate(input=context, max_new_chars=500)
    for gen in batch_gen.tolist():
        print('----\n',dataset.decode(gen))
