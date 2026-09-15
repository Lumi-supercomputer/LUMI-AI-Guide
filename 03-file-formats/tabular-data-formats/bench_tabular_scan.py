"""
Analytical/scan benchmark for parquet vs csv vs hdf5.

Tests three access patterns, in increasing order of how much they should
favor a columnar format:
  1. full scan       — read the entire table
  2. column subset    — read only 2-3 of the columns
  3. filtered read    — read only rows matching a condition (predicate pushdown)

CSV cannot push down either column selection or filtering at the file-read
level, so it always pays the full-scan cost first.
"""
import os
import time
import argparse
import pandas as pd

FORMATS = {
    "parquet": "taxi.parquet",
    "csv": "taxi.csv",
    "hdf5": "taxi.h5",
}


def bench_full_scan(fmt, path):
    t0 = time.time()
    if fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "hdf5":
        df = pd.read_hdf(path, key="taxi")
    return time.time() - t0, len(df)


def bench_column_subset(fmt, path, columns):
    t0 = time.time()
    if fmt == "parquet":
        df = pd.read_parquet(path, columns=columns)
    elif fmt == "csv":
        df = pd.read_csv(path, usecols=columns)
    elif fmt == "hdf5":
        df = pd.read_hdf(path, key="taxi", columns=columns)
    return time.time() - t0, len(df)


def bench_filtered(fmt, path, column, threshold):
    t0 = time.time()
    if fmt == "parquet":
        df = pd.read_parquet(path, filters=[(column, ">", threshold)])
    elif fmt == "csv":
        # No pushdown available in CSV: must read everything, then filter in memory.
        df = pd.read_csv(path)
        df = df[df[column] > threshold]
    elif fmt == "hdf5":
        df = pd.read_hdf(path, key="taxi")
        df = df[df[column] > threshold]
    return time.time() - t0, len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data-formats-tabular")
    parser.add_argument("--columns", nargs="+", default=["trip_distance", "fare_amount"])
    parser.add_argument("--filter-column", default="passenger_count")
    parser.add_argument("--filter-threshold", type=float, default=4)
    args = parser.parse_args()

    print(f"{'format':<10} {'full scan (s)':>15} {'col subset (s)':>16} {'filtered (s)':>14} {'rows(full/sub/filt)':>22}")
    for fmt, fname in FORMATS.items():
        path = os.path.join(args.datadir, fname)
        t_full, n_full = bench_full_scan(fmt, path)
        t_cols, n_cols = bench_column_subset(fmt, path, args.columns)
        t_filt, n_filt = bench_filtered(fmt, path, args.filter_column, args.filter_threshold)
        rows_str = f"{n_full}/{n_cols}/{n_filt}"
        print(f"{fmt:<10} {t_full:15.3f} {t_cols:16.3f} {t_filt:14.3f} {rows_str:>22}")


if __name__ == "__main__":
    main()
