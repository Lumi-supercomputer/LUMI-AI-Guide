# LUMI AI Guide

This repository is a practical guide for moving machine learning training workloads to LUMI using a runnable Vision Transformer example in PyTorch.

All Python and shell scripts referenced in this guide are part of this repository: [LUMI-AI-Guide](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main). The workflow starts from [`1-quickstart/visiontransformer.py`](1-quickstart/visiontransformer.py) and scales up chapter by chapter.

## Goal

Provide a clear core path from first single-GPU execution to distributed training, then optional experiment-tracking extensions on LUMI.

## Requirements

Before starting, ensure you have:

- basic familiarity with Python and machine learning workflows
- a LUMI user account and basic command-line/Slurm usage
- a project with available GPU hours if you want to run the examples

## Important account note

This guide keeps `#SBATCH --account=...` lines inside job scripts for simple copy-run commands.
Before submitting jobs, replace those account values with your own LUMI project account.

## Start here

Begin with [0. How to use this guide](0-how-to-use-guide/README.md). Then follow core lessons `1` to `6`, and continue with optional lessons `7` to `9` if needed.

## Core lessons (recommended order)

- [0. How to use this guide](0-how-to-use-guide/README.md)
- [1. QuickStart](1-quickstart/README.md)
- [2. Setting up your own environment](2-setting-up-environment/README.md)
- [3. File formats for training data](3-file-formats/README.md)
- [4. Data Storage Options](4-data-storage/README.md)
- [5. Multi-GPU and Multi-Node Training](5-multi-gpu-and-node/README.md)
- [6. Monitoring and Profiling jobs](6-monitoring-and-profiling/README.md)

## Optional lessons (experiment tracking)

- [7. TensorBoard visualization](7-tensorboard-visualization/README.md)
- [8. MLflow visualization](8-mlflow-visualization/README.md)
- [9. W&B visualization](9-wandb-visualization/README.md)

## Further reading

- [LUMI Documentation](https://docs.lumi-supercomputer.eu/)
- [LUMI AI Factory Services](https://docs.lumi-supercomputer.eu/software/local/lumi-aif/)
- [LUMI software library, PyTorch](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/)
- [LUMI software library, TensorFlow](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/t/TensorFlow/)
- [LUMI software library, Jax](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/j/jax/)
- [Workshop material - Moving your AI training jobs to LUMI](https://lumi-supercomputer.github.io/LUMI-training-materials/ai-20240529/)

## Navigation

- Next: [0. How to use this guide](0-how-to-use-guide/README.md)
