import json
from glob import glob


def getTime(ops, data):
    # Calculate the total duration of the operation 'ops' from data, arbitrary units
    return sum([i["dur"] for i in data["traceEvents"] if i["name"] == ops]) / 1000000


# It helps to visually analyse the trace on ui.perfetto.dev to identify key operations
ops = [
    "hipMemcpyWithStream",
    "enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__",
    "aten::copy_",
]
for trace in glob("*-trace.json"):
    with open(trace, "r") as fd:
        data = json.load(fd)
    print("Analyzing", trace)
    for op in ops:
        print("Time of", op, ": ", getTime(op, data))
