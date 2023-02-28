import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import nerfstudio.scripts.eval as eval
import tyro
import fileinput
import torch
import gc
import yaml

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
        "--output-name",
        type=str,
        default="metrics",
        help="Directory of the checkpoint relative to model's path",
    )

    args.add_argument(
        "--cameras",
        nargs="+",
        required=True,
        help="Camera jsons to evaluate",
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

    for camera_path in tqdm(args.cameras):
        for checkpoint_path in tqdm(checkpoints):
            camera_path = Path(camera_path)
            checkpoint_path = Path(checkpoint_path)
            parent_dir = (
                checkpoint_path.parent
            )  # models/2023-01-23-23-35-51-za_30-specimen_rgb_n_12-nerfacto
            checkpoint_name = checkpoint_path.stem
            config_path = Path(parent_dir, args.config_reldir, "config.yml")
            config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
            output_directory = Path("renders", camera_path.stem, f"{config.experiment_name}-{checkpoint_name}")
            output_directory.mkdir(parents=True, exist_ok=True)
                        
            eval_args = [
                "--load-ckpt", str(checkpoint_path),
                "--load-config", str(config_path),
                "--traj", "filename", 
                "--camera-path-filename", str(camera_path),
                "--output-format", "images",
                "--output-path", str(output_directory),
            ]
                        
            tyro.cli(RenderTrajectory, args=eval_args).main()

            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
