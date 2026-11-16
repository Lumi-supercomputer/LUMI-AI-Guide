# Tabular Data Formats on LUMI

Every training job on an HPC system computes on the GPU while it retrieves the data to train on. If data can't arrive fast enough, the GPU sits idle. For tabular data, format choice determines whether that data arrives fast or becomes the bottleneck.

This chapter compares three formats, Parquet, CSV and HDF5, on two different ways of reading the same data.

## TLDR

- Scanning, filtering and file size: Parquet was the fastest and the smallest of the three formats.
- Random row batches: CSV and Parquet were both fast once loaded (about 1 s for 10,000 rows), because the benchmark loads the whole table into memory first. HDF5 reads rows from the file on demand; its time varied from 1.9 s to 128 s between runs (median 103 s).
- Parquet is not always the best choice: see [Decision framework](#decision-framework), and note that results depend on your table's size and on how you read it.

## What is being tested

1. **Scanning and filtering**: reading a full table, only specific columns, and reading only rows matching a condition. This is the pattern behind data exploration, feature engineering, and analytical queries.
2. **Random row-batch access**: repeatedly grabbing small, randomly-ordered batches of rows, the way a model's DataLoader pulls training data.

A format can be excellent at one and bad at the other, so testing only one pattern and generalizing to the best format is misleading. The formats tested here also differ in how they retrieve the data. Some hold it in memory, while others perform repeated file access. These differences in data access can affect the read speed; see [What load time and dataloader time measure](#what-load-time-and-dataloader-time-measure).

## Dataset used

NYC Yellow Taxi trip records (one month, ~2.96 million rows, 19 columns), downloaded from the [NYC Taxi and Limousine Commission's public trip record data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

## Formats compared

- **Parquet**: a columnar binary format that stores values column by column. This enables efficient data access by reading only the necessary columns and using row-group statistics to skip irrelevant data during query execution.
- **CSV** (Comma Separated Values): a plain text format with one row per line. It has no internal structure and no compression. It was included as the no-optimization baseline.
- **HDF5** (Hierarchical Data Format): a high-performance binary format for large structured scientific and numerical data.

## Results

Each benchmark was run 5 times. Values are the median in seconds, with the range (min–max) in brackets.

### File size

| Format  | Size     |
|---------|----------|
| Parquet | 58.0 MB  |
| CSV     | 296.2 MB |
| HDF5    | 404.5 MB |

### Scanning and filtering

| Format  | Full scan            | Column subset (2 of 19) | Filtered             |
|---------|----------------------|-------------------------|----------------------|
| Parquet | 0.506 (0.322–0.539)  | 0.023 (0.020–0.037)     | 0.153 (0.120–0.214)  |
| CSV     | 4.993 (3.998–6.337)  | 1.686 (1.612–2.265)     | 4.681 (3.950–5.807)  |
| HDF5    | 8.722 (1.221–13.424) | 1.564 (1.170–1.903)     | 0.348 (0.226–0.435)  |

### Random row batch access (10,000 rows)

| Format  | Load time            | Dataloader time         |
|---------|----------------------|-------------------------|
| Parquet | 1.953 (0.148–2.099)  | 1.004 (0.977–1.988)     |
| CSV     | 5.497 (4.024–9.868)  | 0.811 (0.699–1.479)     |
| HDF5    | 6.003 (0.062–7.676)  | 103.040 (1.902–128.315) |

HDF5's dataloader time varied widely between runs. The two fastest runs also opened the file almost instantly (load time 0.06 s and 0.11 s, against 6.0 to 7.7 s in the others), which suggests the file was already in the node's cache. Read the HDF5 median with this in mind.

## What load time and dataloader time measure

The three formats split their cost between "load" and "dataloader" differently:

- **CSV** reads the entire file into memory during load time, so the dataloader step only looks up rows that are already in memory. All of the cost is paid upfront, which is why CSV has the slowest load time and the fastest dataloader time.
- **HDF5** only opens the file during the load, with each random batch triggering a new file query. As a result, the data access cost is paid repeatedly and this is evident with a higher dataloader time.
- **Parquet** memory-maps the file during load, then decodes it into an in-memory table. Each random batch then slices rows out of that table, so, like CSV, the dataloader step does no file reads.

## Decision framework

Choose a format based on how you plan to read your data:

- If you mostly scan full tables, read specific columns, or filter rows (data exploration, feature engineering): use Parquet. It was roughly 10x faster than CSV for a full scan, and roughly 70x faster than both CSV and HDF5 when reading two columns.
- If you mostly grab small, randomly-ordered batches of rows (feeding a model during training) and the table fits in memory: use Parquet or CSV. CSV in this experiment was only fast because the whole table sits in memory.
- If the table does not fit in memory: HDF5 reads rows from the file on demand instead of loading everything first. Expect slower and less predictable times than the in-memory formats.
- If your table is much larger or differently shaped than the ~2.96 million rows and 19 columns tested here, treat these numbers as directional, and run the benchmarks on your own data (see [BENCHMARKS.md](BENCHMARKS.md)).

Parquet was the fastest and smallest here for scanning, filtering and file size, but it has downsides:
- Random row access: Parquet is optimized for column reads. Fetching whole rows from a file typically requires reading and decoding data from all relevant columns, which can make random row access less efficient than column-focused queries.
- Not human-readable: you can't open a Parquet file in a text editor, and adding or changing rows means rewriting the file or writing new files.
- Needs a library: such as `pyarrow`, whereas almost any tool can read CSV.

## Parquet's row group size

Parquet splits a file into row groups, each storing per-column statistics that let it skip groups that don't match a filter. Smaller groups have more metadata to look through and less efficient compression, while larger groups have more data to read to fetch a few rows.

We generated Parquet files with three different row-group configurations and compared their on-disk size and their read performance on the random row-batch benchmark (3 runs, median in seconds, range in brackets):

| Setting | Row group size (max rows)           | Row groups created | File size | Load time            | Dataloader time      |
|---------|-------------------------------------|--------------------|-----------|----------------------|----------------------|
| small   | 1000                                | 2,965              | 89.9 MB   | 4.069 (0.948–6.464)  | 2.990 (2.885–3.146)  |
| default | 1,048,576 (pyarrow default)         | 3                  | 58.0 MB   | 2.664 (0.164–5.423)  | 1.024 (0.943–1.069)  |
| large   | 2,000,000                           | 2                  | 58.0 MB   | 2.216 (0.173–3.352)  | 1.046 (0.933–1.053)  |

The last row group in each file holds the remaining rows. Small row groups made the file about 55% larger and the dataloader about 3x slower, while default and large behaved almost identically. Keep in mind that, this benchmark loads each file into memory before timing. The dataloader times therefore reflect picking rows out of an in-memory table, and reading row groups directly from disk may behave differently.

It's important to note that your results will depend heavily on specifics of your own data:

- How many rows your table has; in this experiment, our "small" setting created ~3,000 row groups specifically because we have ~3 million rows. A much smaller table might barely be affected by any of these three settings at all.
- How many columns you have, and how wide each row is; the per-row-group summary cost scales with the column count, so a table with fewer or more columns than the 19 in this dataset will see a different balance of costs.
- How you actually plan to read the data; if you only ever scan the whole table in order and never grab random batches, this whole row-group-size question may matter far less to you than it does for a training pipeline.

## Running the benchmarks

Step-by-step instructions for reproducing these results, including the row group experiment, are in [BENCHMARKS.md](BENCHMARKS.md).

## Scope and limitations

These are single-node benchmarks on LUMI's shared Lustre file system at a specific point in time. Results will vary with concurrent cluster load and are not a substitute for multi-node or distributed I/O testing.

### Table of contents
- [Home](../..#readme)
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
