# 4. Data Storage Options

> [!NOTE]  
> If you wish to run the included examples on LUMI, have a look at the [quickstart](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/1-quickstart#readme) chapter for instructions on how to set up the required environment.

This section describes the most useful data storage options for AI users on LUMI.

Machine learning frameworks generally make extensive use of parallel processing to shorten time to obtain results. With the increased number of nodes available for computations, a single storage target might become a bottleneck for I/O, especially when large amounts of data are to be accessed.

To that end, LUMI employs a parallel filesystem, [Lustre](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lustre/), which allows for storage and retrieval of files in a distributed way, using striping to distribute files across multiple Object Storage Targets (OSTs). These can then be accessed from multiple compute nodes in a parallel way, and thus significantly increase I/O performance.

The main available filesystems on LUMI which have high priority for computations requiring high-throughput I/O are [LUMI-P](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lumip/), [LUMI-F](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lumif/) and RAM FS. In the context of massive parallel processing access, [LUMI-O](https://docs.lumi-supercomputer.eu/storage/lumio/) provides generally significantly poorer performance, and so is not considered in this documentation.

## LUMI-P vs LUMI-F vs RAMfs - Overview
### LustreFS
**LUMI-P** and **LUMI-F** have the same software backend (LustreFS), but they differ in achievable bandwidth. LUMI-F provides about 10x higher bandwidth than LUMI-P, but has higher accounting costs (at the time of writing 3x higher) than LUMI-P. When using LUMI-F, you may want to consider moving data to and from LUMI-F before and after periods of computations in your project so that your available TBhours units are not exhausted unnecessarily quickly. The accounting parameters of LUMI-P and LUMI-F are explained in the [Storage billing](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/#storage-billing) section of the LUMI documentation.

### RAMfs
Every compute node makes its RAM available to processes and to RAMfs. This RAMfs can be used for local-only but fast access to your data. This requires that as part of your task you first copy the relevant data into the node's RAMfs available at `/tmp`, and at the end of your task you copy the data you wrote there out of the node. If your job on a node is aborted, any data you did not copy out of the node is lost, as each node is cleaned after a job terminates - for any reason.

The RAM is shared with your tasks, so you have to make sure that your processes and your RAMfs files fit together within the available memory.

On `small-g` partition, only 64GB of ca. 460GB is available for RAMfs unless you allocate the nodes with `--mem 0 --exclusive` SLURM options. On `standard-g` partition, all ca. 460GB of RAM is available for RAMfs storage by default. 

## Making the most of LustreFS 
### Striping
To maximize I/O throughput on LUMI-P or LUMI-F, you want to ensure that your concurrent file access uses different OSTs. If your data files are large enough, they can be striped evenly among OSTs, and you only need to specify the number of stripes you want to use (unless you intend to overstripe, this is equal to the number of available OSTs for a given file). As an example, you can configure striping with `32` stripes in `4MiB` stripe chunks for all files that will be created in `/your/newly/created/data/directory`, or for an `/individual/empty/file` that you will populate with the following command:
``` bash
lfs setstripe --stripe-count 32 --stripe-size 4m /your/newly/created/data/directory /individual/empty/file
```

### Overstriping
An advanced Lustre feature known as 'overstriping' addresses some limitations by allowing a single file to have more than one stripe per OST. The presence of more than one stripe per OST allows the full bandwidth of a given OST to be exploited while still using one file. [This presentation](https://wiki.lustre.org/images/b/b3/LUG2019-Lustre_Overstriping_Shared_Write_Performance-Farrell.pdf) discusses application I/O performance using overstriping and implications for achieving expected performance in shared-file I/O workloads. See slide #10 in the said PDF for a quick graphical representation of overstriping.

## Throughputs in numbers
### LUMI-P
LUMI-P has 32 OSTs, which allow for optimum performance for large files when the stripe count is set to 32 or 64.

The average read/write throughput to an OST in LUMI-P is 161 MB/second for moving data to/from the memory of a compute node. Depending on the current filesystem load, if you have 16 concurrent threads accessing separate chunks of the file, you can achieve 16 * 161 = 2,576 MB/sec throughput.

### LUMI-F
LUMI-F has 58 OSTs, which allow for optimum performance when the stripe count is set to 58 or a multiple of 58.

The average read/write throughput to an OST in LUMI-F is 1,710 MB/second for moving data to/from the memory of a compute node. Depending on the current filesystem load, if you have 16 concurrent threads accessing separate chunks of the file, you can achieve 16 * 1,710 = 27,360 MB/sec throughput.

### RAMfs
If you can segment your data sets such that each node has exclusive access to the data it needs fast access to, you can utilize RAMfs. RAMfs provides the fastest per-node access to your data, but this data is then not natively shared nor shareable among nodes. RAMfs is only visible to its node, where such RAM resides.

For tasks which can make use of a non-distributed filesystem, on a single node, RAM can be effectively utilized through the RAM FS `/tmp` mountpoint. Obviously, RAMfs is the fastest of the storage options, but has the serious drawback of not being distributed, but rather localized to a single node. For completeness we do include the benchmark speeds in the standard-g partition:

The average read from RAM is 7,800 MB/second. The average write into RAM is 2,900 MB/second. Multiple readers/writers do not significantly increase these values, as the local RAM data bus is the bottleneck.

An example of using RAMfs for the visualtransformer script can be found in [run_ramfs.sh](run_ramfs.sh) in tandem with [visualtransformer_ramfs.py](visualtransformer_ramfs.py).

### Further reading
 - [Loading Training data on LUMI](https://462000265.lumidata.eu/ai-20241126/files/LUMI-ai-20241126-10-Training_Data_on_LUMI.pdf)
 - [LUG2019-Lustre_Overstriping_Shared_Write_Performance-Farrell.pdf](https://wiki.lustre.org/images/b/b3/LUG2019-Lustre_Overstriping_Shared_Write_Performance-Farrell.pdf)
 - [Configuring Lustre File Striping](https://wiki.lustre.org/index.php/Configuring_Lustre_File_Striping)

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