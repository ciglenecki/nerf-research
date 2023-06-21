import argparse
import fileinput
import gc
from pathlib import Path

import torch
import tyro
import yaml
from tqdm import tqdm

from nerfstudio.scripts.render import RenderTrajectory


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
        "--cameras",
        nargs="+",
        required=True,
        help="Camera jsons to evaluate",
    )

    args.add_argument(
        "--traj",
        type=str,
        required=True,
        choices=["spiral", "filename", "interpolate", "dataset"],
    )
    args.add_argument(
        "--traj-source",
        type=Path,
        help="Dataset directory or cameras.json",
    )

    return args.parse_args()


def main():
    args = parse_args()

    checkpoints = args.checkpoints
    if len(args.checkpoints) == 1:
        checkpoints_file = str(args.checkpoints[0])
        if not checkpoints_file.endswith(".ckpt") and Path(checkpoints_file).is_file():
            checkpoints = [checkpoint_path.rstrip() for checkpoint_path in fileinput.input(checkpoints_file)]

    for camera_path in tqdm(args.cameras):
        for checkpoint_path in tqdm(checkpoints):
            camera_path = Path(camera_path)
            checkpoint_path = Path(checkpoint_path)
            parent_dir = checkpoint_path.parent  # models/2023-01-23-23-35-51-za_30-specimen_rgb_n_12-nerfacto
            checkpoint_name = checkpoint_path.stem
            config_path = Path(parent_dir, args.config_reldir, "config.yml")
            config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
            output_directory = Path("renders", camera_path.stem, f"{config.experiment_name}-{checkpoint_name}")
            output_directory.mkdir(parents=True, exist_ok=True)

            eval_args = [
                "--load-ckpt",
                str(checkpoint_path),
                "--load-config",
                str(config_path),
                "--camera-path-filename",
                str(camera_path),
                "--output-format",
                "images",
                "--output-path",
                str(output_directory),
                "--traj",
                str(args.traj),
                "--traj-source",
                str(args.traj_dataset),
            ]

            tyro.cli(RenderTrajectory, args=eval_args).main()

            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()

"""
python3 src/render_wrap.py \
--checkpoints models/2023-03-31-16-16-13-cu37-three_circles_a_15_25_45_train_81_val_09/step-000009999.ckpt \
--traj filename \
--cameras data/camera/three_circles_a_15_25_45_first_9_on_path.json \
--output-format images

python3 src/render_wrap.py \
--checkpoints models/2023-03-31-16-50-57-kn93-a_30.00_r_0.93_d_1.60_train_25_val_05/step-000016000.ckpt \
--traj filename \
--cameras data/camera/a_30_00_r_0_93_d_1_on_path_last_not.json \
--output-format images
"""
