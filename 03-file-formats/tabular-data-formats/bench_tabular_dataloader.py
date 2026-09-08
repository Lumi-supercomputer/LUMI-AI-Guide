"""
Random-access row-batch benchmark for parquet vs csv vs hdf5, mirroring the
image-pipeline DataLoader benchmark:

  - Parquet and CSV have no native per-row random lookup. Both are loaded
    fully into memory once (a one-time cost, timed separately below), then
    __getitem__ just indexes into that in-memory table.
  - HDF5 (format="table", written by convert_tabular_formats.py) supports
    genuine per-row random reads via PyTables row-number indexing, without
    loading the full table into memory first.

So this benchmark answers: "if I want to feed random row batches to a model
and my table doesn't fit comfortably in memory, which format actually lets
me avoid paying the full-load cost?" not just "which is fastest overall."

"""
import os
import time
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader


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


class CSVRowDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        t0 = time.time()
        self.df = pd.read_csv(self.path)
        self.load_time = time.time() - t0
        return self

    def __exit__(self, *exc):
        return False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()


class HDF5RowDataset(Dataset):
    def __init__(self, path, key="taxi"):
        self.path = path
        self.key = key

    def __enter__(self):
        t0 = time.time()
        self.store = pd.HDFStore(self.path, mode="r")
        self.n = self.store.get_storer(self.key).nrows
        self.load_time = time.time() - t0  
        return self

    def __exit__(self, *exc):
        self.store.close()
        return False

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        row = self.store.select(self.key, where=f"index in [{idx}]")
        return row.iloc[0].to_dict()

    def __getitems__(self, indices):
        idx_list = ",".join(str(i) for i in indices)
        rows = self.store.select(self.key, where=f"index in [{idx_list}]")
        return [row.to_dict() for _, row in rows.iterrows()]


def _collate_passthrough(batch):
    return batch


def run(fmt, datadir, num_workers, n_samples, batch_size=64):
    if fmt == "parquet":
        ds_ctx = ParquetRowDataset(os.path.join(datadir, "taxi.parquet"))
    elif fmt == "csv":
        ds_ctx = CSVRowDataset(os.path.join(datadir, "taxi.csv"))
    elif fmt == "hdf5":
        ds_ctx = HDF5RowDataset(os.path.join(datadir, "taxi.h5"))
    else:
        raise ValueError(fmt)

    with ds_ctx as ds:
        n = min(n_samples, len(ds))
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=_collate_passthrough,
        )
        t0 = time.time()
        seen = 0
        for batch in loader:
            seen += len(batch)
            if seen >= n:
                break
        elapsed = time.time() - t0

    print(f"{fmt} load time: {ds_ctx.load_time:.4f}")
    print(f"{fmt} dataloader time: {elapsed:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data-formats-tabular")
    parser.add_argument("-n", "--num_workers", type=int, default=0)
    parser.add_argument("-N", "--num_samples", type=int, default=100000)
    parser.add_argument("-ff", "--file_format", choices=["parquet", "csv", "hdf5"], required=True)
    args = parser.parse_args()

    run(args.file_format, args.datadir, args.num_workers, args.num_samples)


if __name__ == "__main__":
    main()
