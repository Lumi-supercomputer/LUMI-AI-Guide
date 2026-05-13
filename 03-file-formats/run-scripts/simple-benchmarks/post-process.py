import argparse
from glob import glob
import numpy as np

# Raw data files can be summarized with CLI 'grep time *tiny*.out'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        choices=["seq", "tiny", "large", "full"],
        help="Slurm output file-string",
    )

    args = parser.parse_args()

    files = glob(f"run-scripts/simple-benchmarks/*{args.data}*.out")
    raw_result = {"HDF5": [], "LMDB": [], "SquashFS": []}
    for file_name in files:
        with open(file_name, "r") as fd:
            lines = [line.rstrip("\n") for line in fd]
            try:
                for line in lines:
                    # get the line from the log that has the execution time
                    if "dataloader time" in line:
                        file_format, _, _, time = line.split(" ")
                        raw_result[file_format].append(float(time))
            except Exception as e:
                continue

    result = {
        "HDF5": {"average": 0, "std": 0, "N": 0},
        "LMDB": {"average": 0, "std": 0, "N": 0},
        "SquashFS": {"average": 0, "std": 0, "N": 0},
    }

    for file_format, times in raw_result.items():
        if len(times) > 0:
            result[file_format]["N"] = len(times)
            result[file_format]["average"] = np.average(times)
            result[file_format]["std"] = np.std(times)

    for file_format, result in result.items():
        for i, value in result.items():
            print(f"{file_format}, {i} = {round(value, 2)}")


if __name__ == "__main__":
    main()
