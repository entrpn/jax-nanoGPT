from flax import linen as nn
import jax
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
    dtype = jnp.float32
    bias: bool = True #bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        assert self.config.n_embd % self.config.n_head == 0

        self.c_attn = nn.Dense(3*self.config.n_embd, use_bias=self.config.bias,dtype=self.config.dtype, kernel_init=nn.initializers.normal(stddev=0.02))
        self.c_proj = nn.Dense(self.config.n_embd, use_bias=self.config.bias,dtype=self.config.dtype, kernel_init=nn.initializers.normal(stddev=0.02/math.sqrt(2 * self.config.n_layer)))
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_dropout = nn.Dropout(self.config.dropout)
    
    def __call__(self, x,mask, train=True):
        B, T, C = x.shape
        qkv = self.c_attn(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(B, T, 3*self.config.n_head, C//self.config.n_head)
        qkv = qkv.transpose(0,2,1,3) # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=1)

        att = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        att = att / math.sqrt(q.shape[-1])
        # TODO - attention_masks for varying input lengths, perhaps use softmax where capability https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.activation.softmax.html
        att = jnp.where(mask == 0, -jnp.inf, att)
        #print("mask att:", att)
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
        self.c_fc = nn.Dense(4*self.config.n_embd, dtype=self.config.dtype, use_bias=self.config.bias, kernel_init=nn.initializers.normal(stddev=0.02))
        self.c_proj = nn.Dense(self.config.n_embd, dtype=self.config.dtype, use_bias=self.config.bias, kernel_init=nn.initializers.normal(stddev=0.02/math.sqrt(2 * self.config.n_layer)))
        self.dropout = nn.Dropout(self.config.dropout)
    
    def __call__(self, x, train=True):
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=not train)
        return x

class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        assert self.config.n_embd % self.config.n_head == 0
        self.ln_1 = nn.LayerNorm(epsilon=1e-05, dtype=self.config.dtype, use_bias=self.config.bias)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(epsilon=1e-05, dtype=self.config.dtype, use_bias=self.config.bias)
        self.mlp = MLP(self.config)

    def __call__(self, x, mask=None, train=True):
        x = x + self.attn(self.ln_1(x),mask=mask, train=train)
        x = x + self.mlp(self.ln_2(x), train=train)
        return x

# Bias init for Dense is zeros by default and ones for LayerNorm
class GPT(nn.Module):
    config: GPTConfig
    def setup(self):
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd, dtype=self.config.dtype, embedding_init=nn.initializers.normal(stddev=0.02))
        self.wpe = nn.Embed(self.config.block_size, self.config.n_embd, dtype=self.config.dtype, embedding_init=nn.initializers.normal(stddev=0.02))
        self.drop = nn.Dropout(self.config.dropout)
        self.h = [Block(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-05, dtype=self.config.dtype, use_bias=self.config.bias)
    
    def __call__(self, idx, targets=None, train=True):
        b, t = idx.shape
        assert t <=self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.expand_dims(jnp.arange(0,t,dtype=jnp.int32),0)
        attn_mask = nn.make_causal_mask(idx, dtype=bool)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb, deterministic=not train)
        for block in self.h:
            x = block(x,attn_mask, train)
        
        x = self.ln_f(x)
        # weight tying 
        logits = self.wte.attend(x)
        return logits
