from model import GPTConfig, GPT, CausalSelfAttention
import jax
from jax import random
import jax.numpy as jnp
import optax
import torch
import numpy as np

import tiktoken
enc = tiktoken.get_encoding("gpt2")

# Seeding for random operations
main_rng = random.PRNGKey(42)

print("Device:", jax.devices()[0])

main_rng, x_rng = random.split(main_rng)
config = GPTConfig(vocab_size=100,n_layer=2, n_embd=32,n_head=8, block_size=20)
config = GPTConfig()
x = random.randint(x_rng, (8, config.block_size, config.n_embd), minval=0,maxval=config.vocab_size)
print(x.shape)
attn = CausalSelfAttention(config)
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = attn.init({'params' : init_rng, 'dropout' : dropout_init_rng}, x, train=True)['params']

main_rng, dropout_init_rng = random.split(main_rng)
out = attn.apply({'params' : params}, x, train=True, rngs={'dropout' : dropout_init_rng})
print('out',out.shape)

gpt = GPT(config)

x = random.randint(x_rng, (8, config.block_size), minval=0,maxval=config.vocab_size)

main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = gpt.init({'params' : init_rng, 'dropout' : dropout_init_rng}, x, train=True)['params']

main_rng, dropout_init_rng = random.split(main_rng)
out = gpt.apply({'params' : params}, x, train=True, rngs={'dropout' : dropout_init_rng})
print('out',out.shape)

loss = optax.softmax_cross_entropy_with_integer_labels(out, x).mean()
print("x:",x)
print("out:",out)
print("loss.shape:",loss.shape)
print('loss:', loss)

def generate(idx, model, params, config, max_new_tokens,temperature=1.0,top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.shape[1] <= config.block_size else idx[:,-config.block_size:]
        logits = model.apply({'params': params},idx_cond,train=False)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = jax.lax.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < v[:, [-1]]] = -jnp.inf
        probs = jax.nn.softmax(logits)
        # https://github.com/numpy/numpy/issues/8317
        probs = probs.astype('float64')
        probs = probs / probs.sum()
        idx_next = np.random.multinomial(1,probs.squeeze().tolist())
        idx_next = jnp.argmax(idx_next)
        idx = jnp.concatenate((idx, jnp.array([[idx_next]])), axis=1)
    return idx


x = random.randint(x_rng, (1, config.block_size), minval=0,maxval=config.vocab_size)

logits = generate(x,gpt,params,config,10)
print("logits: ",logits)
print("decoded logits:",enc.decode(logits.squeeze()))