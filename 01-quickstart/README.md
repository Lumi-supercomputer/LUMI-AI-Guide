# 01. QuickStart

This chapter covers how to set up the environment to run the [`visiontransformer.py`](visiontransformer.py) script on LUMI. 

First, you clone this repository to LUMI via the following command:

```bash
git clone https://github.com/Lumi-supercomputer/LUMI-AI-Guide.git
```

We recommend using your `/project/` or `/scratch/` directory of your project to clone the repository as your home directory (`$HOME`) has a capacity of 20 GB and is intended to store user configuration files and personal data.

Next, navigate to the `LUMI-AI-Guide/quickstart` directory:

```bash
cd LUMI-AI-Guide/01-quickstart
```

We now need to setup the environment if we wish to run the included python scripts. We will use one of the provided PyTorch containers. The fastest way to achieve this is to use the provided script `set_up_environment.sh`:

```bash
./set_up_environment.sh
```

If you receive a permission denied error, you can make the script executable by running:

```bash
chmod +x set_up_environment.sh
```

After the script has finished, you will see now one new file in the `LUMI-AI-Guide/resources/` directory. It is the training dataset in a `hdf5` file format (`train_images.hdf5`).

For this example, we use the [Tiny ImageNet Dataset](https://paperswithcode.com/dataset/tiny-imagenet) which is already transformed into the file system friendly hdf5 format (Chapter [File formats for training data](../03-file-formats/README.md) explains in detail why this step is necessary). Please have a look at the terms of access for the ImageNet Dataset [here](https://www.image-net.org/download.php).

To run the Vision Transformer example, we need to use a batch job script. We provide a batch job script [`run.sh`](run.sh) that you can use to run the [`visiontransformer.py`](visiontransformer.py) script on a single GPU on a LUMI-G node. 
A quickstart to SLURM is provided in the [LUMI documentation](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/slurm-quickstart/). 

To run the provided script yourself, you need to replace the `--account` flag in line 2 of the [`run.sh`](run.sh) script with your own project account. You can find your project account by running the command `lumi-workspaces`.

After you have replaced the `--account` flag, you can submit the job to the LUMI scheduler by running:

```bash
sbatch run.sh
```

Once the job starts running, a `slurm-<jobid>.out` file will be created in the `quickstart` directory. This file contains the output of the job and will be updated as the job progresses.

*Note that we do a `module purge` at the beginning of the script. This will cause some warnings that some modules were not unloaded or could not be reloaded.
It is safe to ignore these warnings at this point.*


The output will show Training Loss and Validation Accuracy values for each epoch, similar to the following:

```bash
Training for 4 epochs in total and then saving trained model.
Starting epoch 1.
Epoch 1, Training Loss: 4.825212168216705
Validation Accuracy: 8.95%
Starting epoch 2.
Epoch 2, Training Loss: 4.165826177024841
Validation Accuracy: 14.41%
Starting epoch 3.
Epoch 3, Training Loss: 3.792399849319458
Validation Accuracy: 18.18%
Starting epoch 4.
Epoch 4, Training Loss: 3.558482360649109
Validation Accuracy: 20.945%
Saving model to vit_b_16_imagenet.pth
```

Congratulations! You have run your first training job on LUMI. The next chapter [Setting up your own environment](../02-setting-up-environment/README.md) will explain in more detail how the environment was set up and how you can set up your own environment for your AI projects on LUMI.

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
