# LUMI AI guide

> [!WARNING]  
> This project is still work in progress and changes are made constantly. For well tested examples have a look at the LUMI AI workshop material: https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop


This guide is designed to assist users in migrating their machine learning applications from smaller-scale computing environments to LUMI. We will walk you through a detailed example of training an image classification model using [PyTorch's Vision Transformer (VIT)](https://pytorch.org/vision/main/models/vision_transformer.html) on the [ImageNet dataset](https://www.image-net.org/).

All Python and bash scripts referenced in this guide are accessible in this [GitHub repository](https://github.com/Lumi-supercomputer/LUMI-AI-example/tree/main). We start with a basic python script, [visualtransformer.py](quickstart/visualtransformer.py), that could run on your local machine and modify it over the next chapters to run it efficiently on LUMI.

Even though this guide uses PyTorch, most of the covered topics are independent of the used machine learning framework. We therefore believe this guide is helpful for all new ML users on LUMI while also providing a concrete example that runs on LUMI.

### Requirements

Before proceeding, please ensure you meet the following prerequisites:

* A basic understanding of machine learning concepts and Python programming. This guide will focus primarily on aspects specific to training models on LUMI.
* An active user account on LUMI and familiarity with its basic operations.
* If you wish to run the included examples, you need to be part of a project with GPU hours on LUMI.

### Table of contents

The guide is structured into the following sections:

- [QuickStart](quickstart/README.md)
- [Setting up your own environment](setting-up-environment/README.md)
- [File formats for training data](file-formats/README.md) 
- [Data Storage Options](data-storage/README.md)
- [Multi-GPU and Multi-Node Training](multi-gpu-and-node/README.md)
- [Monitoring and Profiling jobs](monitoring-and-profiling/README.md)
- [TensorBoard visualization](TensorBoard-visualization/README.md)
- [MLflow visualization](MLflow-visualization/README.md)
- [Wandb visualization](Wandb-visualization/README.md)

### Further reading

- [LUMI Documentation](https://docs.lumi-supercomputer.eu/)
- [LUMI software library, PyTorch](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/)
- [LUMI software library, TensorFlow](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/t/TensorFlow/)
- [LUMI software library, Jax](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/j/jax/)
- [Workshop material - Moving your AI training jobs to LUMI](https://lumi-supercomputer.github.io/LUMI-training-materials/ai-20240529/)
