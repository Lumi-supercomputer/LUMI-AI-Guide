# 2. Setting up your own environment

Machine learning frameworks on LUMI serve as isolated environments in the form of container images with a set of Python packages. LUMI uses the [Singularity](https://docs.sylabs.io/guides/main/user-guide/) (SingularityCE) container runtime. Containers can be seen as encapsulated images of a specific environment including all required libraries, tools and Python packages. Container images can be based on virtually any Linux distribution targeting the host architecture, but they still rely on the host kernel and kernel drivers. This plays a significant role in the case of LUMI.

The motivation for using containers on LUMI is twofold: 

- compatibility with ROCm (GPU runtime) and Slingshot network (inter-node communication), 
- filesystem friendliness (encapsulation helps reduce the overhead on the filesystem from accessing numerous small files).

Note that the first point implies that LUMI's containers are **not portable to other machines**, which is usually expected from containers, as these images are unlikely to run on other systems.

## AI Software Environment

The AI Software Environment by LUMI AI Factory is a comprehensive, ready-to-use containerised stack for AI and machine learning workloads on the LUMI supercomputer. The environment is designed to address the complexity of deploying and maintaining AI/ML software in high-performance computing (HPC) setting.

The documentation of the containers can be found in the [LUMI User Guide](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/).

All build artifacts are publicly available. This includes the full recipe,
Containerfiles, build logs, and the resulting final container images. This
transparent approach enables full customization for special use cases, reuse
on other similar systems, as well as adapting the images to run on cloud
environments.

### Available container images


At the moment each release includes the following container images, each building on the previous one by adding
new major functionality in the following order:

1. `lumi-multitorch-rocm-*`: Starts from Ubuntu base image and adds ROCm
2. `lumi-multitorch-libfabric-*`: Adds libfabric to the ROCm image
3. `lumi-multitorch-mpich-*`: Adds MPICH with GPU support to the libfabric image
4. `lumi-multitorch-torch-*`: Adds PyTorch to the MPICH image
5. `lumi-multitorch-full-*`: Adds selection of AI and ML libraries (e.g., Bitsandbytes, DeepSpeed, Flash Attention, Megatron LM, vLLM) to the PyTorch image

The [releases on GitHub](https://github.com/lumi-ai-factory/laifs-container-recipes/releases) also include full details of the included software of each image.

The name of the container includes a timestamp and version identifier. It is explained in the [releases on GitHub](https://github.com/lumi-ai-factory/laifs-container-recipes/releases).

For users running AI applications based on PyTorch the containers starting with `lumi-multitorch-full-*` are most likely the best starting point. Advanced users can build on intermediate containers to customize to their use cases. We aim to release containers for jax or other software in the future.


### Access to container images

The container images are available from the following locations:

- LUMI supercomputer in directory: `/appl/local/laifs/containers/`
- GitHub releases in [public GitHub repository](https://github.com/lumi-ai-factory/laifs-container-recipes/releases)
- Docker Hub in the [LUMI AI Factory organisation](https://hub.docker.com/u/lumiaifactory)

## Interacting with a containerized environment

The Python environment from an image can be accessed either interactively by spawning a shell instance within a container (`singularity shell` command) or by executing commands within a container (`singularity exec` command).

To inspect which specific packages are included in the images you can use this simple command:

```
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
singularity run $SIF pip list
``` 

## Singularity and Slurm

To run a program inside a container on a GPU node, you need to prepend the singularity command with the `srun` launcher. Please note that multiple srun tasks will spawn independent instances of the same container image. 

We can check whether the selected PyTorch image detects the allocated GPU with the following: 

The command

```
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
srun -A <your-project-id> -p small-g -n 1 --gpus-per-task=1 singularity run $SIF python -c "import torch; print(torch.cuda.device_count())"
```

## `singularity-AI-bindings` module

To give LUMI containers access to the Slingshot network for good RCCL and MPI performance and access to the file system of the working directory, some additional bindings are required. As it can be quite cumbersome to set these bindings manually, we provide a module that does this for you. You can load the module with the following commands:

```
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
```

If you prefer to set the bindings manually, we recommend taking a look at the [Running containers on LUMI](https://lumi-supercomputer.github.io/LUMI-training-materials/ai-20240529/extra_05_RunningContainers/) lecture from the [LUMI AI workshop material](https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop).

## Installing additional Python packages in a container 

You might find yourself in a situation where none of the provided containers contain all Python packages you need. One possible way of adding custom packages not included in the image is to use a virtual environment on top of the conda environment. For this example, we need to add the HDF5 Python package `h5py` to the environment:

```
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
singularity shell $SIF
Singularity> python -m venv h5-env --system-site-packages
Singularity> source h5-env/bin/activate
(h5-env) Singularity> pip install h5py
```

This will create an `h5-env` environment in the working directory. The `--system-site-packages` flag gives the virtual environment access to the packages from the container. Now one can execute a script with and import the `h5py` package. To execute a script called `my-script.py` within the container using the virtual environment, use the additional activation command:

```
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
singularity run $SIF bash -c 'source h5-env/bin/activate && python my-script.py'
```

This approach allows extending the environment without rebuilding the container from scratch every time a new package is added. The drawback is that the virtual environment is disjoint from the container, which makes it difficult to move as the path to the virtual environment needs to be updated accordingly. Moreover, installing Python packages typically creates thousands of small files. This puts a lot of strain on the Lustre file system and might exceed your file quota. 

## Custom images

In theory, you can also bring your own container images or convert images from other registries (DockerHub for instance) to the singularity format. In this case it remains your responsibility to keep the container compatible with LUMI's hardware and system environment. We strongly recommend building your containers on top of the LUMI base images provided. 


## ToDO:
- `singularity exec` vs `singularity run`
- there is a basic compiler toolchain included; note specific compiler commands (`gcc-XX` for specific versions installed).
- [cotainr tool](https://lumi-supercomputer.github.io/LUMI-training-materials/ai-20241126/extra_06_BuildingContainers/)
-  turning the virtual environment directory into a [SquashFS file](https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop/blob/main/07_Extending_containers_with_virtual_environments_for_faster_testing/examples/extending_containers_with_venv.md).
- For more information on SLURM on LUMI, please visit the [SLURM quickstart page in our documentation](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/slurm-quickstart/).


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
