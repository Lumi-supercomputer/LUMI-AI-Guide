"""
Converts NYC taxi parquet data into CSV and HDF5, 
tabular-format comparisons: parquet vs csv vs hdf5.

HDF5 is written with format="table" and data_columns=True so it can do
column-subset reads and filtered (predicate pushdown) reads via pandas'
`where=` clause the fairest comparison against parquet's native support
for the same operations. CSV supports neither, which is expected.
"""
import os
import time
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source parquet file (or glob-expanded list)")
    parser.add_argument("--outdir", default="data-formats-tabular")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading {args.input} ...")
    t0 = time.time()
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns in {time.time()-t0:.2f}s")

    parquet_path = os.path.join(args.outdir, "taxi.parquet")
    df.to_parquet(parquet_path, engine="pyarrow")
    print(f"Wrote {parquet_path}")

    csv_path = os.path.join(args.outdir, "taxi.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    hdf5_path = os.path.join(args.outdir, "taxi.h5")
    df.to_hdf(hdf5_path, key="taxi", mode="w", format="table", data_columns=True)
    print(f"Wrote {hdf5_path}")

    print("\nFile sizes:")
    for p in [parquet_path, csv_path, hdf5_path]:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  {p}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
