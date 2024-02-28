

from imports import (
    Path,math,  torch, F, nn, summary, Enum,
    BaseModel, field_validator, Optional, dataclasses
)


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



# Load the YAML file
llm_config = OmegaConf.load("llm_config.yml")