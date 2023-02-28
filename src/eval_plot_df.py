import argparse
import fileinput
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nerfstudio.scripts.my_utils import get_timestamp


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Checkpoints to evaluate (file or list of checkpoints)",
    )
    args.add_argument(
        "--output",
        default=Path("reports", f"metrics-{get_timestamp()}.csv"),
        help="Checkpoints to evaluate (file or list of checkpoints)",
    )
    return args.parse_known_args()


def main():
    args, _ = parse_args()

    metrics = args.metrics
    if len(metrics) == 1:
        file = str(metrics[0])
        if not file.endswith(".json") and Path(file).is_file():
            metrics = [
                checkpoint_path.rstrip() for checkpoint_path in fileinput.input(file)
            ]
    records = []

    for metric_path in metrics:
        metric = json.load(open(Path(metric_path), "r"))
        record = metric
        records.append(record)

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.output)


if __name__ == "__main__":
    main()
