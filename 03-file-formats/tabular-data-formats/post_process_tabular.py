"""
Compiles bench-scan and bench-dataloader, runs into a single results
table with mean and standard deviation.

Usage: python3 post_process_tabular.py
"""
import re
import statistics as stats
from pathlib import Path
from collections import defaultdict

SCAN_LINE_RE = re.compile(
    r"^(parquet|csv|hdf5)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\S+"
)
LOAD_RE = re.compile(r"(\w+) load time: ([\d.]+)")
DATALOADER_RE = re.compile(r"(\w+) dataloader time: ([\d.]+)")


def summarize(label, values):
    if len(values) < 2:
        return f"{values[0]:.3f}" if values else "N/A"
    mean = stats.mean(values)
    std = stats.stdev(values)
    return f"{mean:.3f} +/- {std:.3f} (n={len(values)})"


def process_scan():
    path = Path("bench-scan-results.log")
    if not path.exists():
        print("No bench-scan-results.log found — run `make bench-scan` at least once.")
        return

    data = defaultdict(lambda: defaultdict(list))
    for line in path.read_text().splitlines():
        m = SCAN_LINE_RE.match(line.strip())
        if m:
            fmt, full, col, filt = m.groups()
            data[fmt]["full"].append(float(full))
            data[fmt]["col"].append(float(col))
            data[fmt]["filt"].append(float(filt))

    print("\n=== Scanning and filtering (seconds) ===")
    print(f"{'format':<10} {'full scan':>20} {'col subset':>20} {'filtered':>20}")
    for fmt in ["parquet", "csv", "hdf5"]:
        if fmt not in data:
            continue
        row = data[fmt]
        print(f"{fmt:<10} {summarize('full', row['full']):>20} "
              f"{summarize('col', row['col']):>20} {summarize('filt', row['filt']):>20}")


def process_dataloader():
    path = Path("bench-dataloader-results.log")
    if not path.exists():
        print("No bench-dataloader-results.log found — run `make bench-dataloader` at least once.")
        return

    text = path.read_text()
    data = defaultdict(lambda: defaultdict(list))
    for fmt, val in LOAD_RE.findall(text):
        data[fmt]["load"].append(float(val))
    for fmt, val in DATALOADER_RE.findall(text):
        data[fmt]["dataloader"].append(float(val))

    print("\n=== Random row-batch access (seconds) ===")
    print(f"{'format':<10} {'load time':>25} {'dataloader time':>25}")
    for fmt in ["parquet", "csv", "hdf5"]:
        if fmt not in data:
            continue
        row = data[fmt]
        print(f"{fmt:<10} {summarize('load', row['load']):>25} "
              f"{summarize('dataloader', row['dataloader']):>25}")


def main():
    process_scan()
    process_dataloader()


if __name__ == "__main__":
    main()
