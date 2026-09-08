# Tabular Data Formats on LUMI

Every training job on a shared HPC system does two things at once: it computes on the
GPU, and it reads data from disk. If data can't arrive fast enough, the GPU sits idle;
expensive hardware, allocated at real cost, doing nothing. For tabular data specifically
(spreadsheet-style rows and columns, sensor logs, trip records, tabular ML features),
format choice determines whether that data arrives fast or becomes the bottleneck.

Format choice is often made by habit "just use CSV" or "just
use Parquet" without checking whether that's actually true for the access pattern being
used.

## What's actually being tested

Not just "which format is fastest" is being looked at but also *how* the data will be 
read. This chapter tests two genuinely different access patterns,since they can favor 
completely different formats:

1. **Scanning and filtering** reading a full table, reading only specific columns, or
   reading only rows matching a condition. This is the pattern behind data exploration,
   feature engineering, and analytical queries.
2. **Random row-batch access** repeatedly grabbing small, randomly-ordered batches of
   rows, the way a model's DataLoader pulls training examples.

Testing only one of these patterns and generalizing to "the best format" is misleading. 
A format can be excellent at one and dramatically bad at
the other.

## Dataset used

NYC Yellow Taxi trip records (one month, ~2.96 million rows, 19 columns), downloaded
directly from the NYC Taxi and Limousine Commission's public data source.

## Formats compared

- **Parquet** 
- **CSV** 
- **HDF5**

## How to run this

```bash
cd tabular-data-formats
make download       
make venv           
make convert       # converts the same source data into all three formats
make bench-scan    # tests full-scan, column-subset, and filtered reads
make bench-dataloader  # tests random row-batch access
```


## Scope and limitations

This is a single-node, single-user benchmark on LUMI's shared Lustre filesystem at a
specific point in time. Results will vary with concurrent cluster load and are not a
substitute for multi-node or distributed I/O testing. The deliverable here is not "the
best tabular format" but a decision framework: given your actual access pattern, which
format is justified.
