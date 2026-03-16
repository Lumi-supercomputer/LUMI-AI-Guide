# 1. QuickStart

This lesson gives you a minimal, end-to-end first run of [`visiontransformer.py`](visiontransformer.py) on LUMI.

## Goal

Run one single-GPU training job on LUMI and confirm that:

- the container works on GPU
- the squashfs extension is available
- training logs and a model checkpoint are produced

## Assumptions

- You have a LUMI project account and can submit jobs to `small-g`.
- The repository is cloned on `/project` or `/scratch` (not `$HOME`).
- `../env.sh` is configured for your environment and points to a valid container via `CONTAINER`.

## Working directory

Run commands in this chapter from:

```bash
cd /path/to/LUMI-AI-Guide/1-quickstart
```

## Minimal run checkpoint

Command:

```bash
sbatch run_base.sh
```

Success signal:

- Job output contains `SMOKE TEST PASSED`.

Clone this repository to LUMI if needed:

```bash
git clone https://github.com/Lumi-supercomputer/LUMI-AI-Guide.git
```

Use `/project` or `/scratch` for this clone. `$HOME` has limited capacity and is meant for configuration and personal files.

Then move to the lesson directory:

```bash
cd LUMI-AI-Guide/1-quickstart
```

The recommended quickstart flow has three steps:

1. Smoke-test the base container.
2. Build the squashfs extension.
3. Run the Vision Transformer training script with that extension.

This keeps the runtime model consistent with the rest of the guide.

## Step 1: Smoke test the base container

Submit the base-container smoke test:

```bash
sbatch run_base.sh
```

Check the job output in:

```bash
slurm-<jobid>.out
```

The output file is written to the directory where you run `sbatch` (here: `1-quickstart/`).

You should see Python/Torch/ROCm version info and `SMOKE TEST PASSED`.

## Step 2: Build the squashfs extension

Build `visiontransformer-env.sqsh` from the base container:

```bash
./build_visiontransformer_sqsh.sh
```

- `../resources/visiontransformer-env.sqsh`

The extension includes packages needed by the sample scripts, such as `h5py`.

## Step 3: Run Vision Transformer

Submit:

```bash
sbatch run_vit.sh
```

`run_vit.sh` uses:

- `env.sh` for container selection
- `../resources/visiontransformer-env.sqsh` for the extended Python environment

For quickstart, `run_vit.sh` uses `torchvision.datasets.FakeData` so the run is independent of dataset preparation.  
Quickstart in this lesson uses FakeData only.
The reported metrics are smoke-test indicators and are not meaningful model-quality numbers.

To run the Vision Transformer example, we use the batch job script [`run_vit.sh`](run_vit.sh), which runs [`visiontransformer.py`](visiontransformer.py) on a single GPU on a LUMI-G node.
A quickstart to SLURM is provided in the [LUMI documentation](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/slurm-quickstart/). 

If needed, replace the `--account` flag in [`run_base.sh`](run_base.sh) and [`run_vit.sh`](run_vit.sh) with your own project account. You can find your project account by running `lumi-workspaces`.

Once the job starts, output is written to:

```bash
slurm-<jobid>.out
```

As in Step 1, this is written in the submit directory (`1-quickstart/`) because the script uses Slurm's default output behavior.

The output will show smoke-test metrics for each epoch, similar to the following:

```bash
Quickstart mode: using FakeData with random labels.
Metrics are only for smoke testing runtime, not model quality.
Starting epoch 1.
Epoch 1, Smoke Loss: 5.82
Smoke Accuracy (chance~0.5%): 0.49%
Starting epoch 2.
Epoch 2, Smoke Loss: 5.46
Smoke Accuracy (chance~0.5%): 0.24%
...
```

## Verify

After the three steps, confirm all of the following:

- Base smoke-test output includes `SMOKE TEST PASSED`.
- `../resources/visiontransformer-env.sqsh` exists.
- Training job output includes epoch-level loss and accuracy logs.
- `vit_b_16_imagenet.pth` is created in `1-quickstart/`.

## Troubleshooting

- Job fails at submission: update `--account` in [`run_base.sh`](run_base.sh) and [`run_vit.sh`](run_vit.sh), then check with `lumi-workspaces`.
- Container variable error (`Set CONTAINER in env.sh`): set `CONTAINER` in `../env.sh` to a valid `.sif`.
- Quickstart data: this lesson uses `FakeData`; no dataset files are required.
- No GPU visible in smoke test: ensure `module load singularity-AI-bindings` is present and rerun.

## Navigation

- Previous: [0. How to use this guide](../0-how-to-use-guide/README.md)
- Next: [2. Setting up your own environment](../2-setting-up-environment/README.md)
