import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_config
import tiktoken


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) 
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  
        queries = self.W_query(x)
        values = self.W_value(x)

    
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) 

        return context_vec



class GPTModel(nn.Module):
    def __init__(self,vocab_size , d_model,seq_len , n_layer , n_head,dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.layers = nn.Sequential(*[TransformerBlock(d_model, n_head, seq_len, dropout) for _ in range(n_layer)])
        self.norm = LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size,bias=False)
        
    def forward(self, x):
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.layers(x)
        x=self.norm(x)
        return self.proj(x)
    
    
class TransformerBlock(nn.Module):
    def __init__(self,d_model, n_head, seq_len, dropout):
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.feedforward = FeedForward(d_model , 4 *d_model)
        self.dropout = nn.Dropout(dropout)
        self.att = MultiHeadAttention(
            d_in=d_model,
            d_out=d_model,
            context_length=seq_len,
            num_heads=n_head,
            dropout=dropout,
            qkv_bias=False)

        
    def forward(self, x):
 
        attention_output = self.att(self.norm_1(x))
        attention_output = self.dropout(attention_output)
        x = x + attention_output
        

        ff_output = self.feedforward(self.norm_2(x))
        ff_output = self.dropout(ff_output)
        x = x + ff_output 
        
        return x
    
    
class LayerNorm(nn.Module):
    def __init__(self, d_model , eps=1e-6):
        super().__init__()
        self.eps=eps
        self.d_model=d_model
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))
      
        
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1 , keepdim=True)
        norms= (x - mean) / (std+ self.eps)
        return self.scale * norms + self.shift
         

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) *
        (x + 0.044715 * torch.pow(x, 3))
    ))
        
class FeedForward(nn.Module):
    def __init__(self,d_in , d_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_in , d_out),
            GELU(),
            nn.Linear(d_out,d_in)
        )
        
    def forward(self,x):
        return self.layers(x)
        

def generate_text(model , idx , max_seq , context_window):
    
    for _ in range(max_seq):
        
        idx_cont = idx[:,-context_window:]
        
        with torch.no_grad():
            logits = model(idx)
            
            logits = logits[:,-1,:]
            
            idx_next = torch.argmax(logits , dim=-1 , keepdim=True)
            
            idx = torch.cat((idx , idx_next) , dim =1)
            
            
    return idx

if __name__ == "__main__":
    
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    batch =[]
    
    text1 = "Every effort moves"
    text2 = "My name is"
    
    batch.append(torch.tensor(tokenizer.encode(text1)))
    batch.append(torch.tensor(tokenizer.encode(text2)))
    
    batch = torch.stack(batch , dim=0)
    
    
    cfg = get_config()
    vocab_size = cfg['vocab_size']
    d_model =cfg['d_model']
    seq_len=cfg['seq_len']
    n_layer=cfg['n_layer']
    n_head = cfg['n_head']
    dropout = cfg['dropout']
    
    model = GPTModel(vocab_size=vocab_size,seq_len=seq_len,d_model=d_model,n_layer=n_layer,n_head=n_head,dropout=dropout)

    out = generate_text(
        model ,
        batch,
        3,
        1
    )
    
    print(out)
    
    print(tokenizer.decode(out[1].tolist()))

    