from imports import Path, torch, F, nn

from char_dataset import CharDataset

class BiGramModel(nn.Module):
    '''
    A Simple Bigram model - Prediction depends only on last char
    '''
    def __init__(self, vocab_size:int, embed_dim:int):
        '''
        Embed_dim - D
        '''
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    
    def forward(self, input, target=None):
        '''
        input - (B,T)
        '''
        # Embedding will replace each int along T with embedding of dim D
        logits = self.embedding(input) # (B,T,D)
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
            logits,_ = self.forward(input=input, target=None) #(B,T,D)
            last_char = logits[:,-1,:] #(B,D)
            last_char_probs = F.softmax(last_char, dim=-1) #(B,D)
            # D represent the probablities of a char in vocab size, here D==len(dataset.chars)
            # So we can construct a prob dist based on last char probabilities and draw a sample from it
            sample_index = torch.multinomial(input=last_char_probs, num_samples=1) #(B,1)
            # Multinomial distribution returns the index of the sample it drew
            # Since our stoi representation is same as intiger index, we can use sample index as it is.
            input = torch.cat([input, sample_index], dim=1)
        return input


if __name__ == '__main__':
    tiny_shakes = Path(__file__).parent/'./tiny_shakespeare.txt'
    dataset = CharDataset(file_path=tiny_shakes)
    model = BiGramModel(vocab_size=len(dataset.chars), embed_dim=len(dataset.chars))
    input, target = dataset.get_batch()
    batch_gen = model.generate(input, max_new_chars=50)
    for gen in batch_gen.tolist():
        print('----\n',dataset.decode(gen))
