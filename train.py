from functools import partial
import argparse
import os
import math
import time
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import jax
import jax.numpy as jnp
import flax
from flax.core.frozen_dict import unfreeze
from flax.training import train_state, checkpoints
from flax.serialization import to_bytes, from_bytes
from flax import jax_utils
from flax.jax_utils import unreplicate
from flax.training.common_utils import shard, shard_prng_key

import optax
import orbax.checkpoint as orbax

from model import GPTConfig, GPT
from configs.shakespeare import config as shakespeare_config
from configs.openwebtext10k import config as openwebtext10k_config

import tiktoken

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

def count_params(params) -> int:
    p = jax.tree_util.tree_map(lambda a: a.size if isinstance(a, jnp.ndarray) else 0, params)
    return jax.tree_util.tree_reduce(lambda a, b: a + b, p)

def train(opt):
    seed = 250
    config = GPTConfig()

    train_config = get_train_config(opt.config)
    model_config = model_config_args[train_config['init_from']]
    config.n_layer = model_config['n_layer']
    config.n_head = model_config['n_head']
    config.n_embd = model_config['n_embd']
    config.block_size = train_config['block_size']

    # Eval
    out_dir = train_config['out_dir']
    log_dir = os.path.join(out_dir,'logs')
    writer = SummaryWriter(log_dir=log_dir)
    eval_interval = train_config['eval_interval']
    eval_iters = train_config['eval_iters']
    best_eval = 1e6

    # Checkpoints
    checkpoint_path = os.path.join(out_dir,'checkpoints')
    os.makedirs(checkpoint_path,exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_path,'weights.msgpack')
    def restore_model(state,model,optimizer, step=0):
        with open(checkpoint_path,"rb") as state_f:
            params = from_bytes(state.params,state_f.read())
        
        state = train_state.TrainState.create(
        apply_fn=model.apply, 
        params=params, 
        tx=optimizer)

        return state

    def save_model(state, step=0):
        print("Created async checkpointer")
        with open(checkpoint_path,"wb") as f:
            state_bytes = to_bytes(state.params)
            f.write(state_bytes)

    # Generate
    gen_interval = train_config['gen_interval']
    def log_generations(text, step):
        writer.add_text("generation",text,global_step=step)

    # Data
    batch_size = train_config['batch_size']
    dataset = train_config['dataset']
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    def get_batch(split, rng, batch_size):
        data = train_data if split == 'train' else val_data
        ix = jax.random.randint(rng, (batch_size,), minval=0, maxval=(len(data) - config.block_size))
        x = jnp.stack([data[i:i+config.block_size] for i in ix],dtype=jnp.int32)
        y = jnp.stack([data[i+1:i+1+config.block_size] for i in ix],dtype=jnp.int32)
        return x, y

    iter_num = 0

    # Training
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    train_batch_size = batch_size * jax.device_count()
    input_shape = (batch_size, config.block_size)
    model = GPT(config)
    main_rng, init_rng, dropout_init_rng = jax.random.split(rng, 3)
    params = jax.jit(model.init)({'params' : init_rng, 'dropout' : dropout_init_rng}, jax.random.randint(init_rng, input_shape, minval=0,maxval=config.vocab_size))
    print("Number of params : ",count_params(params))
    # Optimizer
    # learning rate decay settings
    learning_rate= train_config['learning_rate']
    max_iters = train_config['max_iters']
    lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
    warmup_iters = max_iters // 300 # how many steps to warm up for
    min_lr = learning_rate / 10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    def create_learning_rate_schedule():
        return optax.warmup_cosine_decay_schedule(
                init_value=0,
                peak_value=learning_rate,
                warmup_steps=2000,
                decay_steps=lr_decay_iters,
                end_value=min_lr
            )
    def create_adamw_mask(params, prev_key=None):
        retval = {}
        for key in params.keys():
            val = params[key]
            if isinstance(val, flax.core.frozen_dict.FrozenDict):
                retval[key] = create_adamw_mask(val,key)
            else:
                if "ln_" in key or "bias" in key or "embedding" in key:
                    retval[key] = False
                else:
                    retval[key] = True
        # print(retval)
        return retval

    learning_rate_fn = create_learning_rate_schedule()
    decay_mask = create_adamw_mask(params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.add_decayed_weights(1e-2, mask=decay_mask),
        optax.adamw(
            learning_rate=learning_rate_fn,
            b1=0.9, b2=0.95, eps=1e-8
        )
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, 
        params=unfreeze(params), 
        tx=optimizer)

    del params

    state = jax_utils.replicate(state)


    def temperature_sample(idx, params, config, max_new_tokens,temperature=1.0, top_k=20, rng=jax.random.PRNGKey(0)):
        sampling_loop_init_state = (jnp.array(0), idx, params, rng)
        def select_top_k(tensor, k):
            values, _ = jax.lax.top_k(tensor, k)
            mask = tensor > values.min()
            return mask, jnp.where(mask, tensor, 0.)
        def log(t, eps = 1e-20):
            return jnp.log(t + eps)
        def gumbel_noise(rng, shape):
            noise = jax.random.uniform(rng, shape = shape, minval = 0., maxval = 1.)
            return -log(-log(noise))
        def sampling_loop_cond_fn(state):
            (i, _, _, _) = state
            return i <= max_new_tokens

        def sampling_loop_body_fn(state):
            i, idx, params, rng = state
            rng0, rng1 = jax.random.split(rng)
            model = GPT(config)
            logits = model.apply({'params' : params}, idx, train=False)
            logits = logits[:,-1,:]
            noise = gumbel_noise(rng0, logits.shape)
            if top_k:
                mask, logits = select_top_k(logits, top_k)
                noise *= mask
            
            logits += noise
            sampled_ind = np.argmax(logits, axis = -1)
            idx = jnp.concatenate((idx, jnp.array([sampled_ind])), axis=1)
            idx_cond = idx if idx.shape[1] <= config.block_size else idx[:,-config.block_size:]
            return (i+1, idx_cond, params, rng1)

        final_state = jax.lax.while_loop(sampling_loop_cond_fn, sampling_loop_body_fn, sampling_loop_init_state)
        return final_state[1]

    def eval_step(params, batch):
        inputs, targets = batch
        logits = model.apply({'params' : params}, inputs, train=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss

    def evaluate(rng, p_eval_step, params):
        out = {}
        for split in ['train', 'val']:
            losses = jnp.zeros(eval_iters)
            for k in range(eval_iters):
                rng, input_rng = jax.random.split(rng)
                batch = get_batch(split,input_rng, train_batch_size)
                batch = shard(batch)
                loss = p_eval_step(params, batch)
                losses = losses.at[k].set(loss[0])
            out[split] = losses.mean()
        return out


    def train_step(state, batch, dropout_rng=None):
        inputs, targets = batch
        dropout_rng = jax.random.fold_in(dropout_rng, state.step)

        def compute_loss(params):
            logits = state.apply_fn({'params': params['params']}, inputs, train=True, rngs={'dropout' : dropout_rng})
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
            return loss
        
        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)
        metrics = {"loss" : loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics


    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch")

    dropout_rngs = jax.random.split(rng, jax.local_device_count())
    for _ in range(max_iters):
        rng, input_rng = jax.random.split(rng)

        # Generate single example
        if iter_num % gen_interval == 0 and jax.process_index() == 0:
            print(f" {jax.process_index()} is generating")
            x,_ = get_batch('train',input_rng, 1)
            generation = temperature_sample(x, unreplicate(state).params['params'], config, max_new_tokens=50,temperature=1.0, top_k=20, rng=rng)
            generation = generation.squeeze()
            enc = tiktoken.get_encoding("gpt2")
            generation = enc.decode(generation)
            log_generations(generation, iter_num)
            writer.flush()

        # eval
        if iter_num % eval_interval == 0 and jax.process_index() == 0:
            print(f" {jax.process_index()} is evaluating")
            eval_metrics = evaluate(rng, p_eval_step, state.params['params'])
            train_loss = jax.device_get(eval_metrics['train'])
            val_loss = jax.device_get(eval_metrics['val'])
            writer.add_scalar('train/loss', train_loss,global_step=iter_num)
            writer.add_scalar('val/loss', val_loss,global_step=iter_num)
            lr = jax.device_get(unreplicate(learning_rate_fn(state.step)))
            writer.add_scalar('lr', lr,global_step=iter_num)
            if val_loss < best_eval:
                best_eval = val_loss
                if jax.process_index() == 0:
                    print("saving model")
                    save_model(unreplicate(state), step=iter_num)
                    # Restore checkpoint to validate we can load checkpoints.
                    # Only here for example purposes, don't use as it will reset your state steps to 0 and learning rate.
                    # state = restore_model(unreplicate(state), model, optimizer, step=iter_num)
                    # state = jax_utils.replicate(state)

            writer.flush()
        # Train
        batch = get_batch('train',input_rng, train_batch_size)
        batch = shard(batch)
        state, train_metric = p_train_step(state, batch, dropout_rngs)
        train_metric = unreplicate(train_metric)
        if jax.process_index() == 0:
            print(f" iter {iter_num}, loss : {train_metric['loss']}, lr: {learning_rate_fn(state.step)}")
        iter_num+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config. Ex: shakespeare, openwebtext-10k or openwebtext"
    )
    opt = parser.parse_args()
    train(opt)
    