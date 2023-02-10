# jax-nanoGPT

A replicate [nano-GPT](https://github.com/karpathy/nanoGPT) in JAX.

## Install

Install dependencies

```bash
pip install -r requirements.txt
```

If you want to use this code with TPUs, install:

```bash
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Train single node

To create a dataset run:

```bash
cd data/shakespeare
python prepare.py
```

This will create a train.bin and val.bin which holds GPT2 BPE token ids in one sequence. Now you can train. Go back to the folder with the training script and run.

```bash
python train.py --config shakespeare
```

## Train multi node in GCP cloud

We can scale our training by using [TPU pod slices](https://cloud.google.com/tpu/docs/jax-pods) and TPU-VMs. In short, we deploy multiple workers and execute the training job on each worker and let pmap handle scaling.

1. We'll be using TPU-v4. which requires a subnet in the zone `us-central2-b`. Follow the instructions for [Set up and prepare a Google Cloud project](https://cloud.google.com/tpu/docs/v4-users-guide#project-setup).

1. Create an instance. Change `your_project_id` to yours. 

    ```bash
    export TPU_NAME=tpu-v4
    export ZONE=us-central2-b
    export RUNTIME_VERSION=tpu-vm-v4-base
    export PROJECT_ID=<your_project_id>
    export ACCELERATOR_TYPE=v4-16

    gcloud compute tpus tpu-vm create ${TPU_NAME} \
    --zone us-central2-b \
    --accelerator-type ${ACCELERATOR_TYPE} \
    --version ${RUNTIME_VERSION} \
    --subnetwork=tpusubnet \
    --network=tpu-network
    ```

1. In order to ssh into the machine, you might need to modify ~/.ssh/config. Change <your_user_name> with your computer's use name (echo ~/) add the following:

    ```bash
    Host tpu-v4
    HostName 107.167.173.130
    IdentityFile /Users/<your_user_name>/.ssh/google_compute_engine
    ```

1. As a test try to ssh. If this works, you're ready to move to the next steps.

    ```bash
    gcloud compute tpus tpu-vm ssh tpu-v4 --worker=0 --zone us-central2-b --project $PROJECT_ID
    ```

1. Now we’ll run a training job on multiple machines. First, install jax[tpu], clone the repository on all machines and install dependencies

    ```bash
    gcloud compute tpus tpu-vm ssh tpu-v4 --zone  us-central2-b --project $PROJECT_ID --worker=all --command="pip install 'jax[tpu]>=0.2.16' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

    gcloud compute tpus tpu-vm ssh tpu-v4 --zone  us-central2-b --project $PROJECT_ID --worker=all --command="git clone https://github.com/entrpn/jax-nanoGPT.git"

    gcloud compute tpus tpu-vm ssh tpu-v4 --zone  us-central2-b --project $PROJECT_ID --worker=all --command="pip install -r jax-nanoGPT/requirements.txt"
    ```

1. Generate the dataset in all devices - (TODO : generate data on single drive and mount it to all instances)

    ```bash
    gcloud compute tpus tpu-vm ssh tpu-v4 --zone  us-central2-b --project $PROJECT_ID --worker=all --command="python3 jax-nanoGPT/data/openwebtext-10k/prepare.py"
    ```

1. Kick off training.

    ```bash
    gcloud compute tpus tpu-vm ssh tpu-v4 --zone  us-central2-b --project $PROJECT_ID --worker=all --command="cd jax-nanoGPT; python3 train.py --config openwebtext-10k"
    ```

## Generate

To generate text, use the `generate.py` script with the config that was used for training and the last checkpoint step that was saved.

```bash
python generate.py --config shakespeare --checkpoint-step 7500
```

Tensorboard logs will be stored in out-{dataset-name} with train/eval loss, learning rate and sampled generations.

## Examples

Training with openwebtext10k dataset for 25k steps, where the last 50 characters in the text are generated.

<p align="center">
    <img src="images/owt10k_generated.png"></img>
    <img height="1000" src="images/owt10k.png"></img>
</br>
</br>