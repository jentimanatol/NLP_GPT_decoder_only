
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_padding_mask(token_ids, pad_id):
    return (token_ids == pad_id).unsqueeze(1).unsqueeze(2)

def make_causal_mask(T, device):
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).unsqueeze(0)

def combine_masks(pad_mask, causal_mask):
    return pad_mask | causal_mask

def scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=0.0,training=True):
    d = q.size(-1)
    scores = (q @ k.transpose(-2,-1)) / math.sqrt(d)
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask, -1e9)
    attn = F.softmax(scores, dim=-1)
    if dropout_p>0:
        attn = F.dropout(attn,p=dropout_p,training=training)
    out = attn @ v
    return out,attn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model=128,n_heads=4,dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model=d_model
        self.n_heads=n_heads
        self.d_head=d_model//n_heads
        self.qkv=nn.Linear(d_model,3*d_model,bias=False)
        self.proj=nn.Linear(d_model,d_model,bias=False)
        self.dropout=dropout

    def forward(self,x,attn_mask=None):
        B,T,_=x.shape
        qkv=self.qkv(x)
        q,k,v=qkv.chunk(3,dim=-1)

        def split(t):
            return t.view(B,T,self.n_heads,self.d_head).transpose(1,2)

        q,k,v=split(q),split(k),split(v)

        out,_=scaled_dot_product_attention(q,k,v,attn_mask,self.dropout,self.training)

        out=out.transpose(1,2).contiguous().view(B,T,self.d_model)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self,d_model=128,d_ff=512,dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_model)
        )
    def forward(self,x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self,d_model=128,n_heads=4,d_ff=512,dropout=0.1):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model)
        self.attn=MultiHeadSelfAttention(d_model,n_heads,dropout)
        self.drop1=nn.Dropout(dropout)

        self.ln2=nn.LayerNorm(d_model)
        self.ffn=FeedForward(d_model,d_ff,dropout)
        self.drop2=nn.Dropout(dropout)

    def forward(self,x,mask):
        attn=self.attn(self.ln1(x),mask)
        x=x+self.drop1(attn)
        ff=self.ffn(self.ln2(x))
        x=x+self.drop2(ff)
        return x

class GPTDecoderOnly(nn.Module):
    def __init__(self,vocab_size,pad_id,block_size=128,d_model=128,n_heads=4,n_layers=2,d_ff=512,dropout=0.1):
        super().__init__()
        self.pad_id=pad_id
        self.block_size=block_size
        self.token_emb=nn.Embedding(vocab_size,d_model,padding_idx=pad_id)
        self.pos_emb=nn.Embedding(block_size,d_model)

        self.blocks=nn.ModuleList([
            DecoderBlock(d_model,n_heads,d_ff,dropout)
            for _ in range(n_layers)
        ])

        self.ln=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab_size)

    def forward(self,tokens):
        B,T=tokens.shape
        pos=torch.arange(0,T,device=tokens.device).unsqueeze(0)
        x=self.token_emb(tokens)+self.pos_emb(pos)

        pad_mask=make_padding_mask(tokens,self.pad_id)
        causal=make_causal_mask(T,tokens.device)
        mask=combine_masks(pad_mask,causal)

        for b in self.blocks:
            x=b(x,mask)

        x=self.ln(x)
        return self.head(x)
