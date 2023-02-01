from functools import partial
import os
import math
import time
from tqdm import tqdm
from absl import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key

import optax

from model import GPTConfig, GPT
import tiktoken
enc = tiktoken.get_encoding("gpt2")

seed = 42
rng = jax.random.PRNGKey(seed)
config = GPTConfig()

class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))

# Eval
out_dir = 'out'
log_dir = os.path.join(out_dir,'logs')
writer = SummaryWriter(log_dir=log_dir)
eval_interval = 2000
log_interval = 1
eval_iters = 5
eval_only = False # if True, script exits right after the first eval
best_eval = 1e6

# Checkpoints
checkpoint_path = os.path.join(out_dir,'checkpoints')
def save_model(params, step=0):
    checkpoints.save_checkpoint(ckpt_dir=checkpoint_path, target=params, prefix=f'gpt2',step=step,overwrite=True)

# Generate
gen_interval = 10000
def log_generations(text, step):
    writer.add_text("generation",text,global_step=step)

# Data
batch_size = 8
dataset = "openwebtext-10k"
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
best_val_loss = 1e9

# Training
rng, dropout_rng = jax.random.split(rng)
rng, input_rng = jax.random.split(rng)
rng, init_rng = jax.random.split(rng)
num_epochs = 600
train_batch_size = batch_size * jax.device_count()
per_device_eval_batch_size = int(batch_size)
eval_batch_size = per_device_eval_batch_size * jax.device_count()
input_shape = (batch_size, config.block_size)
model = GPT(config)
main_rng, init_rng, dropout_init_rng = jax.random.split(rng, 3)
x = jax.random.randint(init_rng, (8, config.block_size), minval=0,maxval=config.vocab_size)
params = model.init({'params' : init_rng, 'dropout' : dropout_init_rng}, x, train=True)['params']

# Optimizer
# learning rate decay settings
learning_rate=6e-4
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
max_iters = 600000
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

def create_learning_rate_schedule():
    return optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=learning_rate,
            warmup_steps=warmup_iters,
            decay_steps=lr_decay_iters,
            end_value=min_lr
        )

learning_rate_fn = create_learning_rate_schedule()
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
        learning_rate=learning_rate_fn,
        b1=0.9, b2=0.95, eps=1e-8
    )
)
state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer, dropout_rng=dropout_rng)

rng = jax.random.PRNGKey(seed)
train_rngs = jax.random.split(rng, jax.local_device_count())

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


def train_step(state, batch, train_rng):
    inputs, targets = batch
    dropout_rng, new_train_rng = jax.random.split(train_rng)

    def compute_loss(params):
        logits = model.apply({'params': state.params}, inputs, train=True, rngs={'dropout' : dropout_rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss
    
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)
    metrics = {"loss" : loss}
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    return new_state, metrics, new_train_rng


p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
state = jax_utils.replicate(state)
p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

logging.info("**** Running training ****")
logging.info(f" Instantaneous batch size per device = {per_device_eval_batch_size}")
logging.info(f" Total optimization steps = {max_iters}")

train_time = 0
epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)

for epoch in epochs:
    train_start = time.time()
    rng, input_rng = jax.random.split(rng)
    train_metrics = []
    steps_per_epoch = int(max_iters // num_epochs)
    # train
    for _ in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
        # eval
        if iter_num % eval_interval == 0:
            eval_metrics = evaluate(rng, p_eval_step, state.params)
            train_loss = jax.device_get(eval_metrics['train'])
            val_loss = jax.device_get(eval_metrics['val'])
            writer.add_scalar('train/loss', train_loss,global_step=iter_num)
            writer.add_scalar('val/loss', val_loss,global_step=iter_num)
            lr = jax.device_get(unreplicate(learning_rate_fn(state.step)))
            writer.add_scalar('lr', lr,global_step=iter_num)
            if val_loss < best_eval:
                best_eval = val_loss
                save_model(unreplicate(state).params, step=iter_num)
            writer.flush()
        # Generate single example
        if iter_num % gen_interval == 0:
            x,y = get_batch('train',input_rng, 1)
            generation = generate(x,model,unreplicate(state).params,config,50)
            generation = generation.squeeze()[-50:]
            generation = enc.decode(generation)
            log_generations(generation, iter_num)
            writer.flush()

        # Train
        rng, input_rng = jax.random.split(rng)
        batch = get_batch('train',input_rng, train_batch_size)
        batch = shard(batch)
        state, train_metric, train_rngs = p_train_step(state, batch, train_rngs)
        train_metrics.append(train_metric)
        iter_num+=1
        
    train_time += time.time() - train_start
    train_metric = unreplicate(train_metric)
    epochs.write(
        f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate:"
        f" {learning_rate_fn(state.step)})"
    )
