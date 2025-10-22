# 1. QuickStart

This chapter covers how to set up the environment to run the [`visiontransformer.py`](visiontransformer.py) script on LUMI. 

First, you clone this repository to LUMI via the following command:

```bash
git clone https://github.com/Lumi-supercomputer/LUMI-AI-Guide.git
```

We recommend using your `/project/` or `/scratch/` directory of your project to clone the repository as your home directory (`$HOME`) has a capacity of 20 GB and is intended to store user configuration files and personal data.

Next, navigate to the `LUMI-AI-Guide/quickstart` directory:

```bash
cd LUMI-AI-Guide/1-quickstart
```

We now need to setup the environment if we wish to run the included python scripts. We will use one of the provided PyTorch containers that we extend with additional packages (this step will be explained in more detail in the next chapter [Setting up your own environment](../2-setting-up-environment/README.md)). The fastest way to achieve this is to use the provided script `set_up_environment.sh`:

```bash
./set_up_environment.sh
```

If you receive a permission denied error, you can make the script executable by running:

```bash
chmod +x set_up_environment.sh
```

After the script has finished, you should see now some new files in the `LUMI-AI-Guide/resources/` directory. These files are the PyTorch container (the file ending with `.sif`), a squashFs file containing additional python packages (`visiontransformer-env.sqsh`), the training dataset in a `hdf5` file format (`train_images.hdf5`) and two directories starting with `deepspeed_*` (these directories are needed for running the deepspeed example). 

For this example, we use the [Tiny ImageNet Dataset](https://paperswithcode.com/dataset/tiny-imagenet) which is already transformed into the file system friendly hdf5 format (Chapter [File formats for training data](../3-file-formats/README.md) explains in detail why this step is necessary). Please have a look at the terms of access for the ImageNet Dataset [here](https://www.image-net.org/download.php).

To run the Visual Transformer example, we need to use a batch job script. We provide a batch job script [`run.sh`](run.sh) that you can use to run the [`visiontransformer.py`](visiontransformer.py) script on a single GPU on a LUMI-G node. 
A quickstart to SLURM is provided in the [LUMI documentation](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/slurm-quickstart/). 

To run the provided script yourself, you need to replace the `--account` flag in line 2 of the [`run.sh`](run.sh) script with your own project account. You can find your project account by running the command `lumi-workspaces`.

After you have replaced the `--account` flag, you can submit the job to the LUMI scheduler by running:

```bash
sbatch run.sh
```

Once the job starts running, a `slurm-<jobid>.out` file will be created in the `quickstart` directory. This file contains the output of the job and will be updated as the job progresses. The output will show Loss and Accuracy values for each epoch, similar to the following:

```bash
Epoch 1, Loss: 4.68622251625061
Accuracy: 9.57%
Epoch 2, Loss: 4.104039922332763
Accuracy: 15.795%
Epoch 3, Loss: 3.7419378942489625
Accuracy: 19.525%
Epoch 4, Loss: 3.6926351853370667
Accuracy: 21.265%
...
```

Congratulations! You have run your first training job on LUMI. The next chapter [Setting up your own environment](../2-setting-up-environment/README.md) will explain in more detail how the environment was set up and how you can set up your own environment for your AI projects on LUMI.

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
