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
## Exploring the effect of row group size on Parquet Storage and Performance?

After evaluating different storage formats, we focused on Parquet and investigated 
the impact of row group size on performance and storage efficiency.

### What is a row group

A Parquet file isn't stored as one giant, solid block. It's split internally into
chunks called row groups, think of it like a large stack of paper cut into
several smaller, separate stacks. Each stack also carries its own small summary
of what's inside it (for every column, the minimum and maximum value present in
that stack). That summary is exactly what lets Parquet skip a whole stack
instantly if it can't possibly contain what you're filtering for no need to
open it at all.

**Row group size** simply means: how many rows go into each stack before Parquet
start a new one.
For smaller row groups, they create more metadata and more independently compressed blocks,
which may increase file size but can also allow Parquet to skip irrelevant data 


more precisely during filtered reads. Larger row groups reduce metadata overhead 
and can improve compression, but may require more data to be read when only a small 
portion of the dataset is needed.

The goal of this experiment is therefore to measure the tradeoff between these
effects and answer two practical questions:

1. **Which row-group configuration provides the fastest read performance** for
random row-batch access?
2. **Which configuration uses the least storage space**, and how much storage
overhead is introduced when many small row groups are created?

To investigate this, we generated Parquet files with three different row-group
configurations and compared both their on-disk size and their read performance.


- **small** -->force many small stacks (1,000 rows each)
- **default** --> let Parquet choose its own stack size automatically
- **large** -->force fewer, bigger stacks (500,000 rows each)

### What to expect: the tradeoff between these three

There are two separate costs that can move in different directions depending on
stack size:

- **Storage cost** More stacks means more repeated per-column summaries stored
  throughout the file, plus slightly less efficient compression (each stack is
  compressed on its own, separately from the others). More stacks generally
  means a little more disk space used, even for the exact same data.
- **Compute cost** This depends entirely on *how* you are reading the file. 
  Reading in order (scanning) behaves differently than reading in a random, 
  scattered order. Smaller stacks can help one of these patterns and hurt the 
  other, since there's a real cost to managing many separate stacks versus a real
  cost to opening one very large stack just to get a few rows out of it.

For this dataset which has 2.96-million-row taxi data, once the row groups are applied
This is what it to expect:

| Setting | Row groups created | Rows per group (approx.) |
|---------|--------------------|--------------------------|
| small   | 2,965              | ~999                     |
| default | 3                  | ~988,000                 |
| large   | 6                  | ~494,000                 |

Notice that "default" and "large" both land in the same territory,
while "small" is genuinely different, nearly a thousand times more stacks. 
That's worth keeping in mind before you guess at the result: two of these 
three settings are more similar to each other than you might expect from 
their names alone.

### Why this connects back to our two original tests

- **Scanning and filtering** reading the whole table, or just specific columns,
  or just rows matching a condition. This is the pattern that benefits
  from row groups' per-stack summaries where more stacks can mean more 
  opportunities to skip work entirely, but also more individual stacks to
  manage and check.
- **Random row-batch access** Here, stack size interacts with randomness 
  differently where grabbing one random row from a huge stack means touching a lot 
  of data just to get a little; grabbing it from a tiny stack means touching very 
  little, but you now have  more stacks to organize and jump between.

### Why your own results might look completely different from ours

Its important to note that the will result depends heavily on specifics of your own data 

- **How many rows your table has** In this experiment, our "small" setting created
 ~3,000 stacks specifically because we have ~3 million rows. A much smaller table 
  might barely be affected by any of these three settings at all.
- **How many columns you have, and how wide each row is** the per-stack
  summary cost scales with column count, so a table with far fewer or far more
  columns than the 19 in this dataset will see a different balance of costs.
- **How you actually plan to read the data** if you only ever scan the whole
  table in order and never grab random batches, this whole row-group-size
  question may matter far less to you than it does for a training pipeline.

### Try it yourself

```bash
make convert-rowgroups
make bench-rowgroups
```

`convert-rowgroups` builds all three versions from the same source data.
`bench-rowgroups` runs the same random-batch-access test against all three,
you can see which setting actually wins and by how much.

**Note:** this experiment only covers Parquet. Parquet is the
only one of the three formats here with a tunable setting like this.

## Scope and limitations

This is a single-node, single-user benchmark on LUMI's shared Lustre filesystem at a
specific point in time. Results will vary with concurrent cluster load and are not a
substitute for multi-node or distributed I/O testing. The deliverable here is not "the
best tabular format" but a decision framework: given your actual access pattern, which
format is justified.
