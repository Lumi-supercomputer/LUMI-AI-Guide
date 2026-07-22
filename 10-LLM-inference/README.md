# LLM inference
This chapter describes how to perform Large Language Model (LLM) inference on LUMI using vLLM. [vLLM](https://docs.vllm.ai/en/latest/) is a popular and memory-efficient inference engine for hosting LLMs.  

In this chapter, we will demonstrate how to run a vLLM server with [`Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), and how to use Python scripts for both interactive and batch inference.

This chapter uses a pinned and date-stamped container `lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif` which includes vLLM that is optimised for running on LUMI. 

> **💡 Note for production:** For your own workloads, you can check `/appl/local/laifs/containers/` for a newer container. We recommend pinning to the latest specific date-stamped container rather than using the `latest` symlink to prevent unexpected updates from breaking your scripts.

## Why vLLM?
vLLM is our recommended inference engine primarily due to two innovations:
- **Paged attention:** Efficiently manages KV (key-value) cache memory, allowing for much larger batch sizes, higher throughput and longer context windows.
- **Continuous batching:** Reduces latency by processing new requests as soon as old ones finish, rather than waiting for an entire batch to complete.

## Inference workflows
There are two ways to interact with a model:
| Workflow                  | Description                                                          | VRAM (GPU memory) & Loading Behaviour                                                                                                          | Best for...                                                                 |
|---------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Server-client mode**    | Deploys vLLM as an OpenAI-compatible API server using a Unix socket. | Load once, run many: Weights are loaded into VRAM when the server starts and stay there until the Slurm job ends.                | Interactive testing, checking the model's 'vibe', troubleshooting, or building a chat interface. |
| **Offline (Python) mode** | Uses the LLM class directly within a Python script to process data.  | Load, process, exit: Weights are loaded into VRAM, all prompts are processed in a single high-speed burst, then VRAM is cleared. | High-throughput batch jobs, benchmarking, and processing static files.      |

## The example scripts
In this chapter, we use three distinct Python scripts to demonstrate different ways of interacting with the model:
- [`chat_with_LLM.py`](chat_with_LLM.py): Enables an interactive, back-and-forth dialogue with the model, including chat history.
- [`batched_inference_from_server.py`](batched_inference_from_server.py): Sends hundreds of prompts simultaneously to a running vLLM server for fast dataset processing or benchmarking.
- [`batched_inference_from_Python.py`](batched_inference_from_Python.py): Starts vLLM directly in Python to load the model for fast dataset processing or benchmarking.

## Hardware requirements
To run an LLM, the model must fit entirely in VRAM. The memory required for model weights depends on the number of parameters and the precision at which they are stored.

As a rule of thumb, at half precision (BF16/FP16), you need 2GB of VRAM per 1B parameters, plus 20% overhead for the KV cache and ROCm overhead. For [`Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B):
- **Weights:** 35B parameters × 2 bytes = **70GB**. Note that for [Mixture-of-Experts (MoE)](https://huggingface.co/blog/moe-transformers) models, all the weights are loaded into VRAM, even though only a fraction (3B in our case) is active at a time.
- **KV cache & overhead:** Adding the 20% buffer brings the total to **≈84GB**. Keep in mind that longer context size requires significantly more VRAM for the KV cache.

Since a single LUMI GCD has 64GB, one is not enough, so we use 2 GCDs (128GB total). For a detailed breakdown of different models and [quantisation](https://bentoml.com/llm/model-preparation/llm-quantization) levels, you can use [this VRAM calculator](https://apxml.com/tools/vram-calculator).

## Workflow A: Server-client mode
Use this if you want to keep the model loaded and interact with it multiple times.

### Step 1: Start the vLLM server
The [`start-vllm-server.sh`](start-vllm-server.sh) script asks Slurm for resources (2 GCDs for 2h, 14 CPU cores and 120GB of RAM), handles the environment setup and launches the model. Update your project ID and submit:

``` bash
sbatch start-vllm-server.sh
```

<details>
<summary><b>► Click here to see what the script is doing under the hood</b></summary>

#### What the launch script does
- **AI bindings:** We perform `module purge` and load `lumi-aif-singularity-bindings` to give LUMI containers access to the file system of the working directory.
- **MIOpen cache redirection:** We redirect the cache of MIOpen (AMD's library of deep-learning primitives) to a temporary directory to avoid collisions with other users on the same node. 
- **Storage redirection:** LLM weights can exceed hundreds of gigabytes, far surpassing the 20GB limit of the default `home` directory. To handle this, the script sets the `HF_HOME` environment variable to your project’s `/scratch/` directory.
- **vLLM cache redirection:** We redirect vLLM’s internal cache to your project’s `/scratch/` directory. This prevents the limited storage quota of your `$HOME` directory from being filled by temporary model artifacts.
- **Private communication:** Instead of hosting the server on a standard network port, the script creates a **Unix domain socket** (.sock file). There are two benefits of this approach:
    - **No port collisions:** It avoids the common "Address already in use" error that occurs if another user is using the same port on a shared node.
    - **Enhanced security:** The socket acts as a private gateway, removing the need for an API key. Access is restricted by file permissions and being on the same node (since only users with a job allocation on that node can access it), preventing other LUMI users from using your model instance.

For a deeper dive into the performance and security benefits of Unix domain sockets, see [this technical overview](https://dev.to/kanywst/the-magic-of-sock-why-modern-infrastructure-relies-on-unix-domain-sockets-4ohl). 

#### The execution command
The core of the script is the `srun` command, which launches the container and initialises the server:
``` bash
srun singularity run \
    $SIF \
    vllm serve $MODEL_NAME \
    --tensor-parallel-size $SLURM_GPUS_ON_NODE \
    --uds $SOCKET_FILE \
    --load-format runai_streamer
``` 
**Flags explained:**
- `vllm serve $MODEL_NAME` is the heart of the command that starts our vLLM server.
- `--tensor-parallel-size` tells vLLM how many GCDs to split the model across. We set this to `$SLURM_GPUS_ON_NODE` so it automatically matches our `#SBATCH` request.
- `--uds $SOCKET_FILE` creates the Unix domain socket we discussed earlier and connects the vLLM server to it.
- `--load-format runai_streamer` is a specialised loader that speeds up the transfer of model weights from the parallel file system to the GPUs, significantly reducing loading times for supported models.

</details>

### Step 2: Interact with the server
Interacting with a running vLLM server requires you to be on the same compute node as the server (and its socket file). We do this by 'jumping into' the compute node's shell, which is called **overlapping**.

1. **Monitor the startup:** 
    Find your job ID with `squeue --me`. As soon as your job status is `R` (Running), the model will start loading into VRAM, which takes time. Check the logs and wait for the "Application startup complete" message:

    ```bash
    tail -f slurm-<job-id>.out
    ```

2. **Enter the compute node's shell:** 
    When the server is ready, overlap into the allocated compute node:

    ```bash
    srun --overlap --jobid <job-id> --pty bash
    ```

3. **Set up the environment:**
    Save the long path to the container in the `SIF` variable and load the bindings to let the container 'see' the filesystem:
    ```bash
    export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

    module purge
    module use /appl/local/laifs/modules
    module load lumi-aif-singularity-bindings
    ```

4. **Launch a client script:**
    Now you can run either the interactive chat or the batched-API script:

    - **💬 Option 1: Interactive chat.** Best for having a back-and-forth conversation and quickly checking the model's "vibe" and output format.
        
        ```bash
        singularity run $SIF python chat_with_LLM.py "Qwen/Qwen3.6-35B-A3B"
        ```
        > **ℹ️ NOTE: Why the `httpx` transport in [`chat_with_LLM.py`](chat_with_LLM.py)?**
        > Standard LLM clients expect an `http://localhost:8000` address. Because we use a Unix socket for security and speed on LUMI, we use the `httpx.HTTPTransport(uds=socket_path)` to redirect the library's traffic into that `.sock` file.

    - **🚀 Option 2: Batched API inference.** Best for sending a lot of prompts, receiving the LLM's responses, and tweaking the prompts to run them again without reloading the model.
        ```bash
        singularity run $SIF python batched_inference_from_server.py "Qwen/Qwen3.6-35B-A3B"
        ```    
        *The results will be saved to `results.json`.*

---

## Workflow B: Offline Python mode
Use this for high-throughput batch processing where you don't need interactive access to the model.

### Step 1: Submit the batch job
The [`run-offline-inference.sh`](run-offline-inference.sh) script asks Slurm for resources, handles the environment setup and executes the `batched_inference_from_Python.py` script inside the container. Update your project ID and submit:

```bash
sbatch run-offline-inference.sh
```

### Step 2: Check the results
Since this is a batch job, you won't see the output in your terminal. You can check the job's progress by reading the output file generated by Slurm:
```bash
tail -f slurm-<job-id>.out
```
Once the job finishes, the model's responses will be saved to `results.json`.

---

## Run an offline throughput test
To understand how many tokens per second your setup can handle, you can run an offline benchmark. This sends a burst of requests to vLLM and measures the raw hardware input and output throughput without the overhead of an API server or data serialisation/deserialisation. This throughput test is a standalone job independent of the workflows above. Update your project ID and submit:
```bash
sbatch test-throughput.sh
```

Once you submit the job, you can view the progress and the final throughput metrics (requests/s, total tokens/s, and output tokens/s) in the Slurm output file:
```bash
tail -f slurm-<job-id>.out
```

This script is mostly identical to `start-vllm-server.sh`. The main difference lies in the following command:

```bash
srun singularity run \
    $SIF \
    vllm bench throughput \
    --model $MODEL_NAME \
    --tensor-parallel-size $SLURM_GPUS_ON_NODE \
    --dataset-name sharegpt \
    --num-prompts 1000 \
    --load-format runai_streamer
```

**Flags explained:**
- `vllm bench throughput` runs vLLM in benchmarking mode.
- `--dataset-name sharegpt` is the dataset of prompts from real-world human/LLM conversations that is run through the model.
- `--num-prompts 1000` truncates the long dataset to 1000 entries. 

---

### Table of contents

- [Home](..#readme)
- [01. QuickStart](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/01-quickstart#readme)
- [02. Setting up your own environment](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/02-setting-up-environment#readme)
- [03. File formats for training data](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/03-file-formats#readme)
- [04. Data Storage Options](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/04-data-storage#readme)
- [05. Multi-GPU and Multi-Node Training](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/05-multi-gpu-and-node#readme)
- [06. Monitoring and Profiling jobs](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/06-monitoring-and-profiling#readme)
- [07. TensorBoard visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/07-TensorBoard-visualization#readme)
- [08. MLflow visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/08-MLflow-visualization#readme)
- [09. Wandb visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/09-Wandb-visualization#readme)
- [10. LLM Inference](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/10-LLM-inference#readme)