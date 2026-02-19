# LUMI AI guide

This guide is designed to assist users in migrating their machine learning applications from smaller-scale computing environments to the LUMI supercomputer. We will walk you through a detailed example of training an image classification model using [PyTorch's Vision Transformer (VIT)](https://pytorch.org/vision/main/models/vision_transformer.html) on the [ImageNet dataset](https://www.image-net.org/).

All Python and bash scripts referenced in this guide are accessible in this [GitHub repository](https://github.com/Lumi-supercomputer/LUMI-AI-example/tree/main). We start with a basic python script, [visiontransformer.py](1-quickstart/visiontransformer.py), that could run on your local machine and modify it over the next chapters to run it efficiently on LUMI.

Even though this guide uses PyTorch, most of the covered topics are independent of the used machine learning framework. We therefore believe this guide is helpful for all new ML users on LUMI while also providing a concrete example that runs on LUMI.

> [!IMPORTANT] 
> The [LUMI AI Factory](lumi-ai-factory.eu) will be the main provider of PyTorch containers on LUMI in the future. 
> You can find more details about these containers in the [LUMI AI Factory AI Software Environment documentation](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/).
> This guide will be updated soon to use the new containers. We expect minor changes, but please read the documentation.

### Requirements

Before proceeding, please ensure you meet the following prerequisites:

* A basic understanding of machine learning concepts and Python programming. This guide will focus primarily on aspects specific to training models on LUMI.
* An active user account on LUMI and familiarity with its basic operations.
* If you wish to run the included examples, you need to be part of a project with GPU hours on LUMI.

### Table of contents

The guide is structured into the following sections:

- [1. QuickStart](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/1-quickstart#readme)
- [2. Setting up your own environment](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/2-setting-up-environment#readme)
- [3. File formats for training data](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/3-file-formats#readme)
- [4. Data Storage Options](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/4-data-storage#readme)
- [5. Multi-GPU and Multi-Node Training](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/5-multi-gpu-and-node#readme)
- [6. Monitoring and Profiling jobs](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/6-monitoring-and-profiling#readme)
- [7. TensorBoard visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/7-TensorBoard-visualization#readme)
- [8. MLflow visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/8-MLflow-visualization#readme)
- [9. Wandb visualization](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/9-Wandb-visualization#readme)
  
### Further reading

- [LUMI Documentation](https://docs.lumi-supercomputer.eu/)
- [LUMI AI Factory Services](https://docs.lumi-supercomputer.eu/software/local/lumi-aif/)
- [LUMI software library, PyTorch](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/)
- [LUMI software library, TensorFlow](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/t/TensorFlow/)
- [LUMI software library, Jax](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/j/jax/)
- [Workshop material - Moving your AI training jobs to LUMI](https://lumi-supercomputer.github.io/LUMI-training-materials/ai-20240529/)
