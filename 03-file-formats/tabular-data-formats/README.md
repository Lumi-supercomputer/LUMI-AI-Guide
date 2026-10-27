# Tabular Data Formats on LUMI

Every training job on an HPC system computes on the GPU while it retrieves the data to train on. If data can't arrive fast enough, the GPU sits idle. For tabular data, format choice determines whether that data arrives fast or becomes the bottleneck.

This chapter compares three formats, Parquet, CSV and HDF5, on two different ways of reading the same data.

## TLDR

- Scanning, filtering and file size: Parquet was the fastest and the smallest of the three formats.
- Random row batches: CSV and Parquet were both fast, because the benchmark loads the whole table into memory first. HDF5 was roughly 300x slower, because it queries the file for every batch.
- Parquet is not always the best choice: see [Decision framework](#Decision-framework), and note that results depend on your table's size and on how you read it.

## What is being tested

1. **Scanning and filtering**: reading a full table, only specific columns, and reading only rows matching a condition. This is the pattern behind data exploration, feature engineering, and analytical queries.
2. **Random row-batch access**: repeatedly grabbing small, randomly-ordered batches of rows, the way a model's DataLoader pulls training data.

A format can be excellent at one and bad at the other, so testing only one pattern and generalizing to the best formt is misleading. The formats tested here also differ in how they retrieve the data.Some hold it in the memory, 
while others perform repeated file access. These differences in data access can affect the read speed; see [What load time and dataloader time measure](#What-load-time-and-dataloader-time-measure).

## Dataset used

NYC Yellow Taxi trip records (one month, ~2.96 million rows, 19 columns), downloaded from the [NYC Taxi and Limousine Commission's public trip record data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

## Formats compared

- **Parquet** a columnar binary format that stores values column-by-column. This enables efficient data access by reading only the necessary columns and using row-group statitics to skip irrelevant data during query execution.
- **CSV** (Comma Separated Values):a plain text format with one row per line. It has no internal structure and no compression. It was included as the no-optimization baseline.
- **HDF5** (Hierarchical Data Format): is a high-performance binary format, for large structured scientific and numerical data. 

## Results

Each benchmark was run 3 times. Values are mean ± standard deviation in seconds.

### File size

| Format  | Size     |
|---------|----------|
| Parquet | 58.0 MB  |
| CSV     | 296.2 MB |
| HDF5    | 453.6 MB |

### Scanning and filtering

| Format  | Full scan      | Column subset (2 of 19) | Filtered      |
|---------|----------------|-------------------------|---------------|
| Parquet | 0.329 ± 0.102  | 0.021 ± 0.003           | 0.124 ± 0.005 |
| CSV     | 4.160 ± 0.287  | 1.529 ± 0.091           | 3.651 ± 0.233 |
| HDF5    | 5.510 ± 3.959  | 1.184 ± 0.056           | 1.172 ± 0.057 |

### Random row batch access (10,000 rows)

| Format  | Load time     | Dataloader time  |
|---------|---------------|------------------|
| Parquet | 1.635 ± 0.343 | 0.993 ± 0.052    |
| CSV     | 4.291 ± 0.099 | 0.688 ± 0.003    |
| HDF5    | 3.286 ± 0.280 | 220.007 ± 29.167 |

## What load time and dataloader time measure

The three formats split their cost between "load" and "dataloader" differently:

- **CSV** reads the entire file into memory during load time, so the dataloader step only looks up rows that are already in memory. All of the cost is paid upfront, which is why CSV has the slowest load time and the fastest dataloader time.
- **HDF5** only opens the file during the load, with each random batch triggering a new file query. As a result, the data acess cost is paid repeatedly and this is evident with a higher dataloader time.
- **Parquet** memory-maps the file during load, then decodes it into an in-memory table. Each random batch then slices rows out of that table, so, like CSV, the dataloader step does no file reads.

## Decision framework

Choose a format based on how you plan to read your data:

- If you mostly scan full tables, read specific columns, or filter rows (data exploration, feature engineering): use Parquet. It was fastest roughly 13x faster than CSV for a full scan, and 55x-70x faster for two columns reads, compared with HDF5 AND CSV, respectively.
- If you mostly grab small, randomly-ordered batches of rows (feeding a model during training): use CSV or Parquet, avoid HDF5. CSV in this experiment was only fast because the whole table sits in memory.
- If your table is much larger or differently shaped than ~2.96 million row amd 19-column table tested here, treat these numbers as directional, and run the benchmarks on your own data. (See [BENCHMARKS.md](BENCHMARKS.md)).

Parquet was the best option for scanning, filtering and file size, but not for everything:
- Random row access: Parquet is optimized for column reads. Fetching whole rows from a file typically rquires reading and decoding data from all relevant columns, which can make random row access less efficient than column focussed queries.
- Not human-readable: You can't open a Parquet file in a text editor, and adding or changing rows means rewriting the file or writing new files.
- Needs a library: such as `pyarrow`, whereas almost any tool can read CSV.

## Parquet's row Group size

Parquet splits a file into row groups, each storing per column statistics that lets it skip groups that dont match a filter. Smaller groups have more metadata to look through and less efficient compression, while larger groups have more data to read to fetch a few rows.

We generated Parquet files with three different row-group configurations and compared both their on-disk size and their read performance.

| Setting | Row groups created | Rows per group (approx.) | File size |
|---------|--------------------|--------------------------|-----------|
| small   | 2,965              | ~999                     | 89.9 MB   |
| default | 3                  | ~988,000                 | 58.0 MB   |
| large   | 6                  | ~494,000                 | 58.2 MB   |

Notice that "default" and "large" both land in almost the same territory, while "small" is genuinely different, larger file size with nearly a thousand times more stacks. 

Its important to note that your results will depends heavily on specifics of your own data 

- How many rows your table has; in this experiment, our "small" setting created ~3,000 stacks specifically because we have ~3 million rows. A much smaller table might barely be affected by any of these three settings at all.
- How many columns you have, and how wide each row is; the per-stack summary cost scales with the column count, so a table with fewer or more columns than the 19 in this dataset will see a different balance of costs.
- How you actually plan to read the data; if you only ever scan the whole table in order and never grab random batches, this whole row-group-size question may matter far less to you than it does for a training pipeline.

## Running the benchmarks

Step-by-step instructions for reproducing these results, including the row groups experiment, are in [BENCHMARKS.md](BENCHMARKS.md). 

## Scope and limitations

These are single-node benchmarkS on LUMI's shared Lustre file system at a specific point in time. Results will vary with concurrent cluster load and are not a substitute for multi-node or distributed I/O testing.

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
