import os
import argparse
import jax
from jax import random, lax
import flax.linen as nn
from flax.training import train_state, checkpoints
from flax.serialization import to_bytes, from_bytes
import orbax.checkpoint as orbax
import jax.numpy as jnp
import tiktoken

from model import GPTConfig, GPT
from configs.shakespeare import config as shakespeare_config
from configs.openwebtext10k import config as openwebtext10k_config

print("Device:", jax.devices()[0])

model_config_args = {
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
}

def get_train_config(train_config_name):
    train_config = None
    if train_config_name == 'shakespeare':
        train_config = shakespeare_config
    elif train_config_name == 'openwebtext-10k':
        train_config = openwebtext10k_config
    return train_config

def temperature_sample(idx, sentence_length, params, config, max_new_tokens,temperature=1.0, top_k=20, rng=jax.random.PRNGKey(0)):
    sampling_loop_init_state = (jnp.array(0), idx, sentence_length, params, rng)
    def select_top_k(tensor, k):
        values, _ = lax.top_k(tensor, k)
        mask = tensor > values.min()
        return mask, jnp.where(mask, tensor, 0.)
    def log(t, eps = 1e-20):
        return jnp.log(t + eps)
    def gumbel_noise(rng, shape):
        noise = jax.random.uniform(rng, shape = shape, minval = 0., maxval = 1.)
        return -log(-log(noise))
    def sampling_loop_cond_fn(state):
        (i, _, _, _, _) = state
        return i <= max_new_tokens

    def sampling_loop_body_fn(state):
        i, idx, sentence_length, params, rng = state
        rng0, rng1 = jax.random.split(rng)
        model = GPT(config)
        logits = model.apply({'params' : params}, idx, train=False)
        # Pull from the last token we care about before padding token.
        logits = logits[:,sentence_length-1,:]
        noise = gumbel_noise(rng0, logits.shape)
        if top_k:
            mask, logits = select_top_k(logits, top_k)
            noise *= mask
        
        logits += noise
        sampled_ind = jnp.argmax(logits, axis = -1)
        idx = jax.lax.cond(sentence_length >=config.block_size,
            lambda: jnp.concatenate((idx, jnp.array([sampled_ind])), axis=1)[:,-config.block_size:],
            lambda: idx.at[:,sentence_length].set(sampled_ind))
        return (i+1, idx,sentence_length+1, params, rng1)

    final_state = lax.while_loop(sampling_loop_cond_fn, sampling_loop_body_fn, sampling_loop_init_state)
    return final_state[1]

def main(opt):
    enc = tiktoken.get_encoding("gpt2")

    config = GPTConfig()
    train_config = get_train_config(opt.config)
    model_config = model_config_args[train_config['init_from']]
    config.n_layer = model_config['n_layer']
    config.n_head = model_config['n_head']
    config.n_embd = model_config['n_embd']
    config.block_size = train_config['block_size']

    rng = jax.random.PRNGKey(0)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    model = GPT(config)
    params = model.init({'params' : rng, 'dropout' : dropout_rng}, jax.random.randint(init_rng, (1,config.block_size), minval=0,maxval=config.vocab_size))

    with open(f'./out-{opt.config}/checkpoints/weights.msgpack','rb') as state_f:
        params = from_bytes(params,state_f.read())['params']

    # Our model isn't trained with a pad token, so we'll use the eot token as a pad token. Although this is not ideal since we added eot tokens to the training data, it works for generation.
    pad_token=50256
    encoded_text = enc.encode("When she was a young woman, she really liked going to the library.")
    encoded_text_len = len(encoded_text)
    x = jnp.expand_dims(jnp.pad(jnp.array(encoded_text),(0,max(0,config.block_size - encoded_text_len)), constant_values=pad_token),0)
    max_new_tokens=200
    logits = temperature_sample(x, encoded_text_len, params,config=config, max_new_tokens=max_new_tokens)
    print("logits: ",logits)
    print("decoded logits:",enc.decode(logits.squeeze()[:encoded_text_len+max_new_tokens]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config. Ex: shakespeare, openwebtext-10k or openwebtext"
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Which checkpoint step to load model weights."
    )
    opt = parser.parse_args()
    main(opt)