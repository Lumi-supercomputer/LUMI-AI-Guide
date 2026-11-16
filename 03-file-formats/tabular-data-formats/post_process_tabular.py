"""
Compiles bench-scan and bench-dataloader, runs into a single results
table with the median and the [min-max] range.

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
    median = stats.median(values)
    return f"{median:.3f} [{min(values):.3f}-{max(values):.3f}] (n={len(values)})"


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
    print(f"{'format':<10} {'full scan':>28} {'col subset':>28} {'filtered':>28}")
    for fmt in ["parquet", "csv", "hdf5"]:
        if fmt not in data:
            continue
        row = data[fmt]
        print(f"{fmt:<10} {summarize('full', row['full']):>28} "
              f"{summarize('col', row['col']):>28} {summarize('filt', row['filt']):>28}")


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
    print(f"{'format':<10} {'load time':>28} {'dataloader time':>28}")
    for fmt in ["parquet", "csv", "hdf5"]:
        if fmt not in data:
            continue
        row = data[fmt]
        print(f"{fmt:<10} {summarize('load', row['load']):>28} "
              f"{summarize('dataloader', row['dataloader']):>28}")


def main():
    process_scan()
    process_dataloader()


if __name__ == "__main__":
    main()
