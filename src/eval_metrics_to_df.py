import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import json
import fileinput
import pandas as pd
from nerfstudio.scripts.my_utils import (
    get_sequence_size_from_experiment,
    get_step_from_ckpt_path,
    get_timestamp,
)


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
        # metric_path = fake_metric
        metric = json.load(open(Path(metric_path), "r"))
        checkpoint_path = Path(metric["checkpoint"])
        step = get_step_from_ckpt_path(checkpoint_path)
        experiment_name = str(checkpoint_path.parent.name)
        sequence_size = (
            get_sequence_size_from_experiment(experiment_name)
            if "_n_" in experiment_name
            else None
        )

        record = {}
        record["experiment_name"] = experiment_name
        record["step"] = step
        record["sequence_size"] = sequence_size
        record["checkpoint_path"] = checkpoint_path
        records.append(record)
        record.update(metric["results"])
    df = pd.DataFrame.from_records(records)
    df.to_csv(args.output)


if __name__ == "__main__":
    main()
