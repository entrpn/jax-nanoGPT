from flax import linen as nn
import jax.numpy as jnp
from dataclasses import dataclass
import math

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    dtype = jnp.bfloat16

def new_gelu(x):
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x * 0.044715 * jnp.power(x, 3.0))))

class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        assert self.config.n_embd % self.config.n_head == 0

        self.c_attn = nn.Dense(3*self.config.n_embd)
        self.c_proj = nn.Dense(self.config.n_embd)
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_dropout = nn.Dropout(self.config.dropout)

        self.mask = jnp.tril(jnp.ones((self.config.block_size,self.config.block_size))).reshape(1,1,self.config.block_size,self.config.block_size)
    
    def __call__(self, x, train=True):
        B, T, C = x.shape
        qkv = self.c_attn(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(B, T, self.config.n_head, -1)
        qkv = qkv.transpose(0,2,1,3) # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        att = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
        att = jnp.where(self.mask == 0, -jnp.inf, att)
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = jnp.matmul(att, v)
        y = y.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        y = y.reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y), deterministic=not train)
        return y

class MLP(nn.Module):
    config: GPTConfig
    def setup(self):
        self.c_fc = nn.Dense(4*self.config.n_embd, dtype=self.config.dtype)
        self.c_proj = nn.Dense(self.config.n_embd, dtype=self.config.dtype)
        self.dropout = nn.Dropout(self.config.dropout)
    
    def __call__(self, x, train=True):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=not train)
        return x

class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        assert self.config.n_embd % self.config.n_head == 0
        self.ln_1 = nn.LayerNorm(epsilon=1e-05, dtype=self.config.dtype)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(epsilon=1e-05, dtype=self.config.dtype)
        self.mlp = MLP(self.config)

    def __call__(self, x, train=True):
        print('block, x.shape:',x.shape)
        x = self.ln_1(x)
        print('x.shape',x.shape)
        x = x + self.attn(self.ln_1(x), train=train)
        x = x + self.mlp(self.ln_2(x), train)
        return x

class GPT(nn.Module):
    config: GPTConfig
    def setup(self):
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd, dtype=self.config.dtype)
        self.wpe = nn.Embed(self.config.block_size, self.config.n_embd, dtype=self.config.dtype)
        self.drop = nn.Dropout(self.config.dropout)
        self.h = [Block(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-05, dtype=self.config.dtype)
        self.lm_head = nn.Dense(self.config.vocab_size, dtype=self.config.dtype, use_bias=False)
    
    def __call__(self, idx, targets=None, train=True):
        print('idx.shape:',idx.shape)
        b, t = idx.shape
        assert t <=self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.expand_dims(jnp.arange(0,t,dtype=jnp.int64),0)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        print('tok_emb.shape:',tok_emb.shape)
        print('pos_emb.shape:',pos_emb.shape)
        x = self.drop(tok_emb + pos_emb, deterministic=not train)
        print('x.shape:',x.shape)
        for block in self.h:
            x = block(x,train)
            print('after block for loop x',x.shape)
        
        logits = self.lm_head(x[:,[-1],:])
        return logits
