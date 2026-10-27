"""
Tests whether Parquet's internal "row group" size affects random row-batch
read speed, following up on review feedback that this mechanism 
was worth investigating directly.
"""
import os
import sys
import time
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader


ROW_GROUP_SIZES = {"small": 1000, "default": None, "large": 500000}


def convert(args):
    print(f"Reading {args.input} ...")
    t0 = time.time()
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns in {time.time()-t0:.2f}s")

    os.makedirs(args.outdir, exist_ok=True)
    for label, size in ROW_GROUP_SIZES.items():
        path = os.path.join(args.outdir, f"taxi_rowgroup_{label}.parquet")
        if size is None:
            df.to_parquet(path, engine="pyarrow")
        else:
            df.to_parquet(path, engine="pyarrow", row_group_size=size)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Wrote {path} (row_group_size={size}, {size_mb:.1f} MB)")

    print()
    print("Note: CSV and HDF5 are not included in this experiment.")


class ParquetRowDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        import pyarrow.parquet as pq
        t0 = time.time()
        self.table = pq.read_table(self.path, memory_map=True)
        self.load_time = time.time() - t0
        return self

    def __exit__(self, *exc):
        return False

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, idx):
        return self.table.slice(idx, 1).to_pylist()[0]


def _collate_passthrough(batch):
    return batch


def bench(args):
    for label in ROW_GROUP_SIZES:
        path = os.path.join(args.datadir, f"taxi_rowgroup_{label}.parquet")
        with ParquetRowDataset(path) as ds:
            n = min(args.num_samples, len(ds))
            loader = DataLoader(
                ds, batch_size=64, shuffle=True,
                num_workers=args.num_workers, collate_fn=_collate_passthrough,
            )
            t0 = time.time()
            seen = 0
            for batch in loader:
                seen += len(batch)
                if seen >= n:
                    break
            elapsed = time.time() - t0
        print(f"rowgroup-{label} load time: {ds.load_time:.4f}")
        print(f"rowgroup-{label} dataloader time: {elapsed:.4f}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_convert = subparsers.add_parser("convert")
    p_convert.add_argument("--input", required=True)
    p_convert.add_argument("--outdir", default="data-formats-tabular")

    p_bench = subparsers.add_parser("bench")
    p_bench.add_argument("--datadir", default="data-formats-tabular")
    p_bench.add_argument("-n", "--num_workers", type=int, default=0)
    p_bench.add_argument("-N", "--num_samples", type=int, default=100000)

    args = parser.parse_args()
    if args.command == "convert":
        convert(args)
    elif args.command == "bench":
        bench(args)


if __name__ == "__main__":
    main()
