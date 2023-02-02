# jax-nanoGPT

A replicate [nano-GPT](https://github.com/karpathy/nanoGPT) in JAX. This is still a work in progress and not complete.

## Install

Install dependencies

```bash
pip install -r requirements.txt
```

If you want to use this code with TPUs, install:

```bash
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Usage

To create a dataset run:

```bash
cd data/shakespeare
python prepare.py
```

This will create a train.bin and val.bin which holds GPT2 BPE token ids in one sequence. Now you can train. Go back to the folder with the training script and run.

```bash
python train.py --config shakespeare
```

Tensorboard logs will be stored in out-{dataset-name} with train/eval loss, learning rate and sampled generations.