"""
TODO matej, finish this implementation
"""
import argparse
import json
import math
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
        "--dataset-dir",
        type=Path,
        help="Directory which contains images and transforms.json",
    )

    args.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory",
        default=Path("data", "_indices"),
    )
    args.add_argument(
        "--train-size",
        metavar="float",
        type=float,
        help="Fraction or int",
    )
    args.add_argument(
        "--val-size",
        metavar="float",
        type=float,
        help="Fraction or int",
    )
    args.add_argument(
        "--trim",
        nargs=2,
        metavar="float",
        type=lambda x: float(x) > 0 and float(x) <= 1.0,
        help="Percentage of trim size",
    )
    args.add_argument(
        "--methods",
        metavar="method name",
        nargs="+",
        type=Method,
        choices=Method,
        help="Subsampling method",
        default=[Method.DISPERSE_LOOP],
    )

    return args.parse_known_args()


def main():

    args, unknown_args = parse_args()
    dataset_dir = args.dataset_dir
    train_size, val_size = args.train_size, args.val_size
    out_dir = args.out_dir
    Path(out_dir).mkdir(exist_ok=True)

    images_dir = Path(dataset_dir, "images")
    image_filenames = [
        p.relative_to(images_dir).as_posix()
        for p in Path(images_dir).glob("**/*")
        if p.suffix in {".png", ".jpeg", ".jpg"}
    ]
    image_filenames = list(sorted(image_filenames))

    sequence_size = len(image_filenames)
    trim_start, trim_end = args.trim or (0, 0)
    trim_start = int(trim_start * sequence_size)
    trim_end = int(trim_end * sequence_size)
    image_filenames = image_filenames[trim_start : sequence_size - trim_end]

    zfill_size = len(str(len(image_filenames)))
    zfill_size = 2 if zfill_size == 1 else zfill_size

    if val_size < 1:
        val_size = int(val_size * len(image_filenames))
    else:
        val_size = int(val_size)

    if train_size < 1:
        train_size = int(train_size * len(image_filenames))
    else:
        train_size = int(train_size)

    for method in args.methods:
        if method == Method.DISPERSE:
            linspace_num = val_size
        elif method == Method.DISPERSE_LOOP:
            linspace_num = val_size + 1

        val_indices = np.rint(np.linspace(0, len(image_filenames) - 1, linspace_num)).astype(int)[:-1]

        train_indices = np.setdiff1d(np.arange(len(image_filenames)), val_indices)
        train_mask = np.rint(np.linspace(0, len(train_indices) - 1, train_size)).astype(int)
        train_indices = train_indices[train_mask]

        # Create the indices dictionary
        indices = {}
        for idx in val_indices:
            indices["images/" + image_filenames[idx]] = "val"
        for idx in train_indices:
            indices["images/" + image_filenames[idx]] = "train"

        # Write indices.json

        indices_name = f"train_{str(train_size).zfill(zfill_size)}_val_{str(val_size).zfill(zfill_size)}.json"
        indices_path = Path(out_dir, indices_name)
        with indices_path.open("w") as f:
            json.dump(indices, f, indent=4)

        print(f"Generated {indices_path}")


if __name__ == "__main__":

    main()
