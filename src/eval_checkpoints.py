import argparse
import fileinput
import gc
from pathlib import Path

import nerfstudio.scripts.eval as eval
import torch
import tyro
from tqdm import tqdm


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--checkpoints",
        nargs="+",
        # required=True,
        help="Checkpoints to evaluate (file or list of checkpoints)",
    )
    args.add_argument(
        "--config-reldir",
        type=Path,
        default=Path("."),
        help="Directory of the checkpoint relative to model's path",
    )

    args.add_argument(
        "--output-name",
        type=str,
        default="metrics",
        help="Directory of the checkpoint relative to model's path",
    )

    return args.parse_known_args()


def main():
    args, _ = parse_args()

    checkpoints = args.checkpoints
    if len(args.checkpoints) == 1:
        checkpoints_file = str(args.checkpoints[0])
        if not checkpoints_file.endswith(".ckpt") and Path(checkpoints_file).is_file():
            checkpoints = [
                checkpoint_path.rstrip()
                for checkpoint_path in fileinput.input(checkpoints_file)
            ]

    for checkpoint_path in tqdm(checkpoints):
        checkpoint_path = Path(checkpoint_path)
        parent_dir = (
            checkpoint_path.parent
        )  # models/2023-01-23-23-35-51-za_30-specimen_rgb_n_12-nerfacto
        checkpoint_name = checkpoint_path.stem
        config_path = Path(parent_dir, "config.yml")
        metrics_path = Path(parent_dir, f"{args.output_name}-{checkpoint_name}.json")

        # if metrics_path.is_file() and metrics_path.stat().st_size >= 29 :  # skip existing metric
        #     print("Skipping", str(metrics_path))
        #     continue

        eval_args = [
            "--load-ckpt",
            str(checkpoint_path),
            "--load-config",
            str(config_path),
            "--output-path",
            str(metrics_path),
        ]
        print(eval_args)
        tyro.cli(eval.ComputePSNR, args=eval_args).main()

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
