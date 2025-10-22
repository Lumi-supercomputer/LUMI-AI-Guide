# 5. Multi-GPU and Multi-Node Training

> [!NOTE]  
> If you wish to run the included examples on LUMI, have a look at the [quickstart](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/1-quickstart#readme) chapter for instructions on how to set up the required environment.

Training Deep Learning models is a resource-intensive task. When the compute and memory resources of a single GPU no longer suffice to train your model, multi-GPU and multi-node solutions can be leveraged to distribute your training job over multiple GPUs or nodes. Various strategies exist to distribute Deep Learning workloads, and various frameworks exist that implement those strategies. In this section, we cover two popular methods: data-parallelism using PyTorch's Distributed Data-Parallel (DDP) module and a mix of data parallelism and model sharding using the DeepSpeed library. We describe the necessary changes to the source code and how to launch the distributed training jobs on LUMI.

## PyTorch DDP

PyTorch DDP can be used to implement data-parallelism in your training job. Data-parallel solutions are particularly useful when you would like to speed up the training process and your model fits in the memory of a single GPU. For example, when you are training on a large dataset.

### Source code changes
The script in [ddp_visiontransformer.py](ddp_visiontransformer.py) implements PyTorch DDP on the visiontransformer example. The following changes to the source code are necessary:

Initialize the distributed environment:

```python
import torch.distributed as dist

dist.init_process_group(backend='nccl')
```

Read the local rank from the LOCAL_RANK environment variable.
```python
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
```

Wrap the model:
```python
from torch.nn.parallel import DistributedDataParallel

model = DistributedDataParallel(model, device_ids=[local_rank])
```

Change the dataloader to use the distributed sampler:

```python
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32, num_workers=7)
```

### Running the DDP example
The distributed training job can be launched in multiple ways. We cover two methods: `torchrun` and `srun`.

#### `torchrun`
##### Single-node, multi-GPU
The jobscript to run the PyTorch DDP example on a single LUMI-G node with all 4 GPUs (8 GCDs) is [run_ddp_torchrun.sh](run_ddp_torchrun.sh). We reserve the full node and launch a single task, with 56 cpus per task:

```bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
```

We use the `torchrun` launcher, which will launch 8 processes on the node:

```bash
srun singularity exec $SIF bash -c '$WITH_CONDA && source visiontransformer-env/bin/activate && python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 ddp_visiontransformer.py'
```

##### Multi-node
The jobscript to run the PyTorch DDP example on 4 full LUMI-G nodes is [run_ddp_torchrun_4.sh](run_ddp_torchrun_4.sh).
To run on multiple nodes, we adjust the job requirements:

```bash
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
```
We set the environment variables that will be used for the distributed initialization:

```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
```

And run with `torchrun`, passing the `--rdzv_*` parameters to the launcher:
```bash
srun singularity exec $CONTAINER bash -c '$WITH_CONDA && source visiontransformer-env/bin/activate && python -m torch.distributed.run --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" ddp_visiontransformer.py'
```

#### srun
##### Single-node, multi-GPU
The jobscript to run the PyTorch DDP example on a single LUMI-G node with all 4 GPUs (8 GCDs) is [run_ddp_srun.sh](run_ddp_srun.sh). Since we launch all processes through srun, we now set ntasks-per-node to 8, and cpus-per-task to 7. We again reserve the full node:

```bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
```

Torch Distributed uses a number of environment variables to initialize the distributed environment. We set these variables:

```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
#export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE
```

Then we run as follows:
```bash
srun singularity exec $CONTAINER bash -c "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID \
                                                                                $WITH_CONDA && source visiontransformer-env/bin/activate && \
                                                                                python ddp_visiontransformer.py"
```
Note that the `RANK` and `LOCAL_RANK` environment variables are exported inside the container and cannot be exported in the Slurm script, as they are only available inside the Slurm jobstep (after srun has launched the process).

##### Multi-node
The jobscript to run the PyTorch DDP example on 4 full LUMI-G nodes is [run_ddp_srun_4.sh](run_ddp_srun_4.sh).
To run on multiple nodes, we only need to adjust the job requirements:

```bash
#SBATCH --nodes=4
```

The environment variables that will be used for the distributed initialization (`MASTER_ADDR` and `MASTER_PORT`) are already set and do not need to be passed to the launcher.


## DeepSpeed

DeepSpeed implements a strategy for distributed training that mixes data parallelism with sharding of model parameters. It supports various levels of model sharding and offloading of parameters to CPU memory. DeepSpeed is particularly useful when your training job does not fit in the memory of a single GPU and you would like to scale your training job to multiple GPUs and/or nodes to leverage the increased combined memory capacity, as well as speed up the training job.

### Source code changes
The script in [ds_visiontransformer.py](ds_visiontransformer.py) implements DeepSpeed on the visiontransformer example. The following changes to the source code are necessary:


Parse command-line parameters:

```python
import argparse

parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
```

Initialize the distributed environment:
```python
import deepspeed

deepspeed.init_distributed()
```

Initialize the DeepSpeed engine:
```python
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=model.parameters())
```

Modify the training loop (and similar for the validation loop):
```python
for images, labels in train_loader:
	images, labels = images.to(model_engine.local_rank), labels.to(model_engine.local_rank)
	optimizer.zero_grad()

	outputs = model_engine(images)
	loss = criterion(outputs, labels)

	model_engine.backward(loss)
	model_engine.step()
	running_loss += loss.item()

```

Change the dataloader to use the distributed sampler:

```python
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32, num_workers=7)
```

### ds_config file
The file ds_config.json contains the DeepSpeed configuration parameters. Some important parameters are:
- zero_optimization: the sharding strategy used by DeepSpeed
- fp16 / bf16: use of half precision


```json
{
  "zero_optimization": {
    "stage": 1
  },

  "fp16": {
    "enabled": false
  }
}
```


### Running the DeepSpeed example
In this example, we use the `torchrun` launcher to launch the DeepSpeed example. It is also possible to launch the job using `srun`, in similar fashion as for the PyTorch DDP example.

#### `torchrun`
##### Single-node, multi-GPU
The jobscript to run the DeepSpeed example on a single LUMI-G node with all 4 GPUs (8 GCDs) is [run_ds_torchrun.sh](run_ds_torchrun.sh). We reserve the full node and launch a single task, with 56 cpus per task:

```bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
```

DeepSpeed uses a number of environment variables to initialize the distributed environment. We set these variables:

```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
```

We use the `torchrun` launcher, which will launch 8 processes on the node:
```bash
srun singularity exec $CONTAINER bash -c 'export CXX=g++-12; $WITH_CONDA && source visiontransformer-env/bin/activate && python -m torch.distributed.run --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT ds_visiontransformer.py --deepspeed --deepspeed_config ds_config.json'
```

##### Multi-node
The jobscript to run the DeepSpeed example on 4 full LUMI-G nodes is [run_ds_torchrun_4.sh](run_ds_torchrun_4.sh).
To run on multiple nodes, we adjust the job requirements:

```bash
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
```

And pass the `--rdzv_*` parameters to the launcher:
```bash
srun singularity exec $CONTAINER bash -c 'export CXX=g++-12; $WITH_CONDA && source visiontransformer-env/bin/activate && python -m torch.distributed.run --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --node_rank $SLURM_PROCID --rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" ds_visiontransformer.py --deepspeed --deepspeed_config ds_config.json'
```


#### `srun`
##### Single-node, multi-GPU
The jobscript to run the DeepSpeed example on a single LUMI-G node with all 4 GPUs (8 GCDs) is [run_ds_srun.sh](run_ds_srun.sh). Since we launch all processes through `srun`, we now set ntasks-per-node to 8, and cpus-per-task to 7. We again reserve the full node:

```bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
```

DeepSpeed uses a number of environment variables to initialize the distributed environment. We set these variables:

```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
#export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE
```

Then we run as follows:
```bash
srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity exec $CONTAINER bash -c 'export CXX=g++-12; export RANK=$SLURM_PROCID; export LOCAL_RANK=$SLURM_LOCALID; $WITH_CONDA && source visiontransformer-env/bin/activate && python ds_visiontransformer.py --deepspeed --deepspeed_config ds_config.json'
```
Note that the `RANK` and `LOCAL_RANK` environment variables are exported inside the container and cannot be exported in the Slurm script, as they are only available inside the Slurm jobstep (after srun has launched the process).

##### Multi-node
The jobscript to run the DeepSpeed example on 4 full LUMI-G nodes is [run_ds_srun_4.sh](run_ds_srun_4.sh).
To run on multiple nodes, we only need to adjust the job requirements:

```bash
#SBATCH --nodes=4
```


## CPU-GPU binding
For optimal performance on a LUMI-G node, it is important to set the correct bindings between CPU cores and GCDs (see also https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/#gpu-binding). We illustrate how this can be achieved for two scenarios: when using the torchrun launcher, as well as `srun`. We use the PyTorch DDP example, but the steps are the same for the DeepSpeed example.

#### torchrun
When torchrun is used, there is no way to pass binding information to the launcher, so the GPU binding has to be set in the python script itself, as follows:

```python
def set_cpu_affinity(local_rank):
    LUMI_GPU_CPU_map = {
        # A mapping from GCD to the closest CPU cores in a LUMI-G node
        # Note that CPU cores 0, 8, 16, 24, 32, 40, 48, 56 are reserved for the
        # system and not available for the user
        # See https://docs.lumi-supercomputer.eu/hardware/lumig/
        0: [49, 50, 51, 52, 53, 54, 55],
        1: [57, 58, 59, 60, 61, 62, 63],
        2: [17, 18, 19, 20, 21, 22, 23],
        3: [25, 26, 27, 28, 29, 30, 31],
        4: [1, 2, 3, 4, 5, 6, 7],
        5: [9, 10, 11, 12, 13, 14, 15],
        6: [33, 34, 35, 36, 37, 38, 39],
        7: [41, 42, 43, 44, 45, 46, 47],
    }
    cpu_list = LUMI_GPU_CPU_map[local_rank]
    print(f"Rank {rank} (local {local_rank}) binding to cpus: {cpu_list}")
    psutil.Process().cpu_affinity(cpu_list)

dist.init_process_group(backend='nccl')

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
rank = int(os.environ["RANK"])
set_cpu_affinity(local_rank)
```

Note that this binding is specific to LUMI-G nodes and may not be optimal (or work) on other systems. Moreover, since the binding is implemented in the training script itself, this is not a portable solution. Using `srun` to launch the training job provides a more portable solution to GPU binding.

#### `srun`
When `srun` is used, Slurm binding options can be used in the job script:

```bash
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

srun --cpu-bind=mask_cpu=$CPU_BIND_MASKS singularity exec $CONTAINER bash -c "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID \
                                                                                $WITH_CONDA && source visiontransformer-env/bin/activate && \
                                                                                python ddp_visiontransformer.py"
```

To output the binding information, `--cpu-bind=v` can be passed to `srun`:
```bash
srun --cpu-bind=mask_cpu=$CPU_BIND_MASKS,v singularity exec ...
```

Since the bindings are not set in the python script but in the job submissions script, this is a more portable solution than what can be achieved with the torchrun launcher.


### RCCL environment variables
In all job scripts, two environment variables should be set to make sure that [RCCL](https://rocm.docs.amd.com/projects/rccl/en/latest/) uses the correct interfaces:

```bash
# To have RCCL use the Slingshot interfaces:
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

# To have RCCL use GPU RDMA:
export NCCL_NET_GDR_LEVEL=PHB
```

On a LUMI-G node, each GPU is connected to a 200Gb/s Network Interface Card (NIC) (see also https://docs.lumi-supercomputer.eu/hardware/lumig/). To make use of this connection, the environment variable `NCCL_NET_GDR_LEVEL` needs to be set to `PHB`. This variable determines the maximum distance between the NIC and the GPU for which GPU Direct RDMA is used. If this variable is set incorrectly, this could result in slower communication between GPUs on different nodes. Note that from ROCm 6.2 onwards, `PHB` is the default value of `NCCL_NET_GDR_LEVEL`.

`NCCL_SOCKET_IFNAME` must be set to make RCCL use the Slingshot-11 interconnect to which each GPU is connected. If this is not set, RCCL will try to use a network interface that it has no access to and inter-node GPU-to-GPU communication will not work.

### Table of contents

- [Home](..#readme)
- [1. QuickStart](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/1-quickstart#readme)
- [2. Setting up your own environment](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/2-setting-up-environment#readme)
- [3. File formats for training data](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/3-file-formats#readme)
- [4. Data Storage Options](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/4-data-storage#readme)
- [5. Multi-GPU and Multi-Node Training](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/5-multi-gpu-and-node#readme)
- [6. Monitoring and Profiling jobs](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/6-monitoring-and-profiling#readme)
- [7. TensorBoard visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/7-TensorBoard-visualization#readme)
- [8. MLflow visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/8-MLflow-visualization#readme)
- [9. Wandb visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/9-Wandb-visualization#readme)
