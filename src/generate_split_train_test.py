"""
TODO matej, finish this implementation
"""

import argparse
from enum import Enum
from pathlib import Path

from nerfstudio.nerfstudio.defaults import SPLIT_INDICES_PATH


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
    args.add_argument("--out-dir", type=bool, default=False, help="Desc")
    args.add_argument(
        "--train-size",
        metavar="float",
        type=float,
        help="Between zero and one",
    )
    args.add_argument(
        "--val-size",
        metavar="float",
        type=float,
        help="Between zero and one",
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
        default=Method.DISPERSE_LOOP,
    )

    return args.parse_known_args()


def main():

    args, unknown_args = parse_args()
    dataset_dir = args.dataset_dir
    train_size, val_size = args.train_size, args.val_size
     
    
    images_dir = Path(dataset_dir, "images")
    image_filenames = (
        p.resolve()
        for p in Path(images_dir).glob("**/*")
        if p.suffix in {".png", ".jpeg", ".jpg"}
    )

    sequence_size = len(image_filenames)
    if args.train_size < 1:
        

if __name__ == "__main__":

    main()
