# 0. How to use this guide

## Goal

Start with a clear plan for running this guide on LUMI, including prerequisites, expected outcomes, and recommended lesson order.

## Assumptions

- You are new to this guide and want a reliable first path through it.
- You can access LUMI and have a project account with GPU allocation.
- You are comfortable with basic Linux shell commands.

## Who this guide is for

This guide is intended for users who want to move model training from smaller environments to LUMI with practical, runnable examples.

## Prerequisites

Before starting lesson 1, ensure you have:

- a LUMI user account
- a project with GPU hours
- basic familiarity with Python and Slurm
- a working clone of this repository in `/project` or `/scratch`

## Recommended learning path

Follow this order for the core workflow (`1` to `6`):

1. [1. QuickStart](../1-quickstart/README.md)
2. [2. Setting up your own environment](../2-setting-up-environment/README.md)
3. [3. File formats for training data](../3-file-formats/README.md)
4. [4. Data Storage Options](../4-data-storage/README.md)
5. [5. Multi-GPU and Multi-Node Training](../5-multi-gpu-and-node/README.md)
6. [6. Monitoring and Profiling jobs](../6-monitoring-and-profiling/README.md)

After lesson `6`, the core path is complete. Continue with optional experiment-tracking modules as needed:

- [7. TensorBoard visualization](../7-tensorboard-visualization/README.md)
- [8. MLflow visualization](../8-mlflow-visualization/README.md)
- [9. W&B visualization](../9-wandb-visualization/README.md)

## Baseline conventions used across lessons

- Keep a single stable container path in `env.sh`.
- Keep datasets in a consistent location under `resources/` unless a chapter explicitly changes it.
- Run batch jobs from the chapter directory so relative paths in scripts resolve correctly.

## What success looks like

By the end of the core workflow, you should be able to:

- run a single-GPU training job
- scale to multi-GPU and multi-node runs
- choose storage and data formats intentionally
- profile runs and interpret bottlenecks

## Troubleshooting

- Job submission fails immediately: verify `--account` and project permissions.
- Container-related errors: confirm `env.sh` points to an existing `.sif`.
- Missing files at runtime: run commands from the intended chapter directory.

## Navigation

- Previous: [Guide Home](../README.md)
- Next: [1. QuickStart](../1-quickstart/README.md)
