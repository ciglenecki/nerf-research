"""
TODO matej, finish this implementation
"""

import argparse
from pathlib import Path

from nerfstudio.nerfstudio.defaults import SPLIT_INDICES_PATH


class Method(Enum):
    FIRST_LAST_INTERPOLATE = "first_last"
    RANDOM = "random"
    TAILS = "tails"
    MIDDLE = "middle"

    def __str__(self):
        return self.value
    
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--filename", type=Path, default=SPLIT_INDICES_PATH)
    args.add_argument("--image-dir", type=Path, help="Desc")
    args.add_argument("--dataset-dir", type=Path, help="Desc")
    args.add_argument("--outdir", type=bool, default=False, help="Desc")
    args.add_argument(
        "--frac",
        metavar="float",
        type=lambda x: float(x) > 0 and float(x) <= 1.0,
        help="Between zero and one",
    )
    args.add_argument(
        "--methods",
        metavar="method name",
        nargs="+",
        type=Method,
        choices=Method,
        help="Subsampling method",
        default=Method.DISPERSE,
    )
    return args.parse_known_args()

def main():
    args, unknown_args = parse_args()
    
    if int(bool(args.dataset_dir)) + int(bool(args.image_dir)) != 1:
        raise Exception("Provide either --image-dir or --dataset-dir")
    
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir, "images")
    else:
        dataset_dir = args.dataset_dir
    
    
    
if __name__ == "__main__":
    
    main()