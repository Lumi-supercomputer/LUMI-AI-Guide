# Running the Tabular Format Benchmarks

These steps reproduce the results in the [tabular data formats README](README.md).

## Main benchmarks

1. Replace the account `project_xxxxxxxxx` in `Makefile` with your project id.
2. Run `make download` to download the raw taxi data.
3. Run `make venv` to install packages not included in the base container.
4. Run `make convert` to convert the same source data into Parquet, CSV and HDF5.
5. Run `make bench-scan` to test full-scan, column-subset and filtered reads.
6. Run `make bench-dataloader` to test random row-batch access.
7. Run `make post` to compile the results. It prints two tables, one for scanning and
   filtering and one for random row-batch access, each showing the median, the [min-max] range and
   the number of runs for each format.

Run steps 5 and 6 several times (the README results use 5 runs each). Times can vary a lot
between runs, in particular depending on whether the files are already cached on the node,
so compare medians and look at the range.

## Row group experiment

After `make download` and `make venv`:

```bash
make convert-rowgroups
make bench-rowgroups
```

`convert-rowgroups` writes three Parquet versions of the same table (small, default and large
row groups; large has more rows per group than the default). `bench-rowgroups` runs the random
row-batch test on each of them and prints the load time and dataloader time. Run it several
times (the README results use 3 runs).
