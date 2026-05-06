<h1 align="center"> LUMI AI guide </h1>

<p align="center">
  <img src="assets/images/LUMI_logo.jpg" alt="LUMI Logo"/>
</p>

This guide is designed to assist users in migrating their machine learning applications from smaller-scale computing environments to the LUMI supercomputer. We will walk you through a detailed example of training an image classification model using [PyTorch's Vision Transformer (VIT)](https://pytorch.org/vision/main/models/vision_transformer.html) on the [ImageNet dataset](https://www.image-net.org/).

All Python and bash scripts referenced in this guide are accessible in this [GitHub repository](https://github.com/Lumi-supercomputer/LUMI-AI-example/tree/main). We start with a basic python script, [visiontransformer.py](1-quickstart/visiontransformer.py), that could run on your local machine and modify it over the next chapters to run it efficiently on LUMI.

Even though this guide uses PyTorch, most of the covered topics are independent of the used machine learning framework. We therefore believe this guide is helpful for all new ML users on LUMI while also providing a concrete example that runs on LUMI.

> [!IMPORTANT] 
> PyTorch containers on LUMI will in the future be provided by the [LUMI AI Factory](lumi-ai-factory.eu). 
> This guide, apart from [03. File formats for training data](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/03-file-formats#readme), was updated to utilize these new containers. The containers currently referenced in that part remain available on LUMI, but will no longer receive updates. However, all examples included in this guide will continue to work as they currently do.
> For more information about the new containers, refer to the [LUMI AI Factory AI Software Environment documentation](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/).

### Requirements

Before proceeding, please ensure you meet the following prerequisites:

* A basic understanding of machine learning concepts and Python programming. This guide will focus primarily on aspects specific to training models on LUMI.
* An active user account on LUMI and familiarity with its basic operations.
* If you wish to run the included examples, you need to be part of a project with GPU hours on LUMI.

### Table of contents

The guide is structured into the following sections:

- [01. QuickStart](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/01-quickstart#readme)
- [02. Setting up your own environment](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/02-setting-up-environment#readme)
- [03. File formats for training data](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/03-file-formats#readme)
- [04. Data Storage Options](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/04-data-storage#readme)
- [05. Multi-GPU and Multi-Node Training](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/05-multi-gpu-and-node#readme)
- [06. Monitoring and Profiling jobs](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/06-monitoring-and-profiling#readme)
- [07. TensorBoard visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/07-TensorBoard-visualization#readme)
- [08. MLflow visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/08-MLflow-visualization#readme)
- [09. Wandb visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/09-Wandb-visualization#readme)
  
### Further reading

- [LUMI Documentation](https://docs.lumi-supercomputer.eu/)
- [LUMI AI Factory Services](https://docs.lumi-supercomputer.eu/software/local/lumi-aif/)
- [LUMI AI software environment, PyTorch](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/)
- [LUMI software library, TensorFlow](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/t/TensorFlow/)
- [LUMI software library, Jax](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/j/jax/)
- [Workshop material - Moving your AI training jobs to LUMI](https://lumi-supercomputer.github.io/AI-latest)

### Primary contributors
The primary contributors to this repository are the [LUMI User Support Team (LUST)](https://lumi-supercomputer.eu/user-support/need-help/
) and the [LUMI AI Factory](https://lumi-ai-factory.eu).

[![image](assets/images/LUMI_AIF_logo.png)](https://lumi-ai-factory.eu)
[![image](assets/images/LUMI_AIF_funding.png)](https://lumi-ai-factory.eu)