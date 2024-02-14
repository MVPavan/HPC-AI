from imports import (
    Path,math,  torch, F, nn, summary, Enum,
    BaseModel, field_validator, Optional, dataclasses
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

class GPT2Choice(str, Enum):
    gpt2 = 'gpt2'
    gpt2_medium = 'gpt2-medium'
    gpt2_large = 'gpt2-large'
    gpt2_xl = 'gpt2-xl'
    custom = 'custom'
    
class AttBase(BaseModel):
    vocab_size:int = 50304 # GPT2 Vocab Size
    embed_dim:int = 64
    n_heads:int = 4
    n_blocks:int = 4
    context_length:int = 32
    bias:bool = False
    mlp_hidden_dim:Optional[int] = None
    model_choice:str = GPT2Choice.custom

class AttParams(AttBase):    
    batch:int = 16
    device:str = 'cuda'
    lr:float = 1e-3
    dropout:float = 0.0
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
    
    def model_post_init(self, *args, **kwargs) -> None:
        # When n_heads concatenated use linear layer to project to -> embed_dim
        if self.model_choice == GPT2Choice.gpt2:
            self.n_blocks, self.n_heads, self.embed_dim = 12, 12, 768
        elif self.model_choice == GPT2Choice.gpt2_medium:
            self.n_blocks, self.n_heads, self.embed_dim = 24, 16, 1024
        elif self.model_choice == GPT2Choice.gpt2_large:
            self.n_blocks, self.n_heads, self.embed_dim = 36, 20, 1280
        elif self.model_choice == GPT2Choice.gpt2_xl:
            self.n_blocks, self.n_heads, self.embed_dim = 48, 25, 1600
        
        if not self.model_choice == GPT2Choice.custom:
            self.vocab_size = 50257
            self.context_length = 1024
            self.bias = True

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

class MultiHeadAttention(nn.Module):
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
        self.linear_dropout = nn.Dropout(params.dropout)
        self.params = params

    def forward(self, x:torch.Tensor, causal=False):
        # X - B,T,D
        # Out - B,T,H (H = D//N)
        B,T,D = x.shape

        q,k,v = self.w_kqv(x).split(D, dim=2) # B,T,3*D -> split -> 3(B,T,D)
        q,k,v = [m.view(B,T,params.n_heads,params._head_dim).transpose(1,2) for m in (q,k,v)] # 3(B,N,T,H)

        if self.flash:
            out = F.scaled_dot_product_attention(
                query=q,key=k,value=v,
                attn_mask=None,
                dropout_p=self.params.dropout if self.training else 0,
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
        out = self.linear_dropout(self.proj(out))
        return out # B,T,D

class MLP(nn.Module):
    '''
    Multi layer perceptron
    '''
    def __init__(self, params: AttParams):
        super().__init__()
        factor = 4 if params.mlp_hidden_dim is None else params.mlp_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features=params.embed_dim, out_features=factor*params.embed_dim, bias=params.bias),
            nn.GELU(),
            nn.Linear(in_features=factor*params.embed_dim, out_features=params.embed_dim, bias=params.bias),
            nn.Dropout(params.dropout)
        )
    
    def forward(self, x):
        # x -> B,T,D
        return self.mlp(x) # B,T,D

class AttentionBlock(nn.Module):
    '''
    Collection of LayerNorm, Multihead attention, Residual, LayerNorm, FeedForward, Residual
    '''
    def __init__(self, params: AttParams):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=params.embed_dim)
        self.mha = MultiHeadAttention(params=params)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=params.embed_dim)
        self.mlp = MLP(params=params)

    def forward(self,x):
        out = self.layer_norm_1(x)  # B,T,D
        out = x + self.mha(x)  # B,T,D
        out = self.layer_norm_2(out)  # B,T,D
        out = x + self.mlp(out)  # B,T,D
        return out  # B,T,D

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
        self.decoder = nn.ModuleDict(dict(
            token_embedding = nn.Embedding(num_embeddings=params.vocab_size, embedding_dim=params.embed_dim),
            pos_embedding = nn.Embedding(num_embeddings=params.context_length, embedding_dim=params.embed_dim),
            input_drop = nn.Dropout(params.dropout),
            mha_list = nn.ModuleList([AttentionBlock(params) for _ in range(params.n_blocks)]),
            final_layer_norm = nn.LayerNorm(normalized_shape=params.embed_dim)
        )) # B,T,D
        self.llm_head = nn.Linear(in_features=params.embed_dim, out_features=params.vocab_size, bias=False) # B,T,V
        self.decoder.token_embedding.weight = self.llm_head.weight # https://paperswithcode.com/method/weight-tying
        self.params = params

        self.apply(self.__init_weights)
        self.__gpt2_inits()

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def __gpt2_inits(self):
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.params.n_blocks))

    @staticmethod
    def key_map(k:str):
        k = k.replace('transformer','decoder')
        k = k.replace('transformer','token_embedding')
        k = k.replace('transformer','pos_embedding')

    @classmethod
    def from_pretrained(cls, params: AttParams):
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(params.model_choice, token="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")

        model = GPT(params=params)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        sd_hf = model_hf.state_dict()
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        # for k in sd_keys_hf:
        #     if any(k.endswith(w) for w in transposed):
        #         # special treatment for the Conv1D weights we need to transpose
        #         assert sd_hf[k].shape[::-1] == sd[k].shape
        #         with torch.no_grad():
        #             sd[k].copy_(sd_hf[k].t())
        #     else:
        #         # vanilla copy over the other parameters
        #         assert sd_hf[k].shape == sd[k].shape
        #         with torch.no_grad():
        #             sd[k].copy_(sd_hf[k])
        
        for k, hf_k in zip(sd_keys, sd_keys_hf):
            if any(hf_k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[hf_k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[hf_k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[hf_k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[hf_k])

        return model, model_hf
        

    def forward(self, input, target=None):
        '''
        input - (B,T)
        '''
        # Embedding will replace each int along T with embedding of dim D
        B,T = input.shape
        token_embeds = self.decoder.token_embedding(input) # (B,T,D)
        position_embeds = self.decoder.pos_embedding(torch.arange(end=T, dtype=torch.long, device=token_embeds.device)) # T,D
        x = self.decoder.input_drop(token_embeds+position_embeds) # B,T,D + T,D (Broadcasting) -> B,T,D
        for att_block in self.decoder.mha_list:
            x = att_block(x) # B,T,D
        x = self.decoder.final_layer_norm(x)
        logits = self.llm_head(x) # B,T,V
        if target is None:
            return logits, None

        # cross entropy expects V or channel to be second dim
        B,T,V = logits.shape
        logits = logits.view(B*T,V)
        target = target.view(B*T)
        loss = F.cross_entropy(input=logits,target=target, ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, input, max_new_chars:int=10, temperature:float=1.0, top_k:Optional[int]=None):
        '''
        input - (B,T)
        '''
        for _ in range(max_new_chars):
            # Crop input to max context length
            logits,_ = self.forward(input=input[:,-self.params.context_length:], target=None) #(B,T,V)
            last_char = logits[:,-1,:] #(B,V)
            
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # Return min(k,v) largest values
                logits[logits < v[:, [-1]]] = -float('Inf') # after softmax all these will become zero
            
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
    # tiny_shakes = Path(__file__).parent/'./tiny_shakespeare.txt'
    # dataset = CharDataset(file_path=tiny_shakes)
    # params = AttParams(vocab_size=len(dataset.chars))
    params = AttParams(model_choice=GPT2Choice.gpt2_medium)
    # model = GPT(params=params)
    model, model_hf = GPT.from_pretrained(params=params)
    print("Here")

    # x,y = dataset.get_batch(context_length=params.context_length, batch=params.batch)
    # model_summary = summary(model=model,input_data=(x,y))
    # trainer = Trainer(model=model, dataset=dataset, params=params)
    # trainer.train()

    # context = torch.zeros((1, 1), dtype=torch.long, device=params.device)
    # batch_gen = trainer.model.generate(input=context, max_new_chars=500)
    # for gen in batch_gen.tolist():
    #     print('----\n',dataset.decode(gen))
