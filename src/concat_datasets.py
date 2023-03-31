"""
TODO matej, finish this implementation
"""
import argparse
import json
import math
import shutil
from enum import Enum
from pathlib import Path

import numpy as np


class Method(Enum):
    DISPERSE = "disperse"
    DISPERSE_LOOP = "disperse_loop"

    def __str__(self):
        return self.value


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        help="Datasets",
    )

    args.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory",
    )
    args.add_argument(
        "--use-subdir",
        type=Path,
        default="image",
        help="Output directory",
    )
    return args.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir
    use_subdir = args.use_subdir
    Path(out_dir).mkdir(exist_ok=True)
    final_seq = {}
    num_images = 0
    zfill = 5
    for dataset in args.datasets:
        images_dir = Path(dataset, "images")
        image_filenames = [p for p in Path(images_dir).glob("**/*") if p.suffix in {".png", ".jpeg", ".jpg"}]
        image_filenames = list(sorted(image_filenames))
        for image_filename in image_filenames:
            ext = image_filename.suffix
            frame_name = f"frame_{str(num_images).zfill(zfill)}"
            dest = Path(out_dir, f"{frame_name}{ext}")
            print(image_filename)
            print(dest)
            print()
            shutil.copyfile(image_filename, dest)
            num_images += 1


if __name__ == "__main__":

    main()
