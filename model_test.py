from model import GPTConfig, GPT, CausalSelfAttention
import jax
from jax import random

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
