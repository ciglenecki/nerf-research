import argparse
from pathlib import Path
from enum import Enum
import numpy as np
import operator
import shutil
from tqdm import tqdm
from shutil import copytree, ignore_patterns


class Method(Enum):
    FIRST_LAST_INTERPOLATE = "first_last"
    DISPERSE = "disperse"

    # RANDOM = "random"

    def __str__(self):
        return self.value


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--input", type=Path, nargs="+", required=True, help="Input directories"
    )
    args.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory. Subdirectories will be created automatically.",
    )
    args.add_argument(
        "--fractions",
        metavar="floats",
        nargs="+",
        type=float,
        help="Shares of sequence to generate",
        default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )

    args.add_argument(
        "--ns",
        metavar="ints",
        nargs="+",
        type=int,
        help="Exact number of images in the sequence",
        default=[8, 7, 6, 5, 4, 3, 2],
    )
    args.add_argument(
        "--methods",
        metavar="method name",
        nargs="+",
        type=Method,
        choices=list(Method),
        help="Subsampling method",
        default=[Method.DISPERSE],
    )

    return args.parse_known_args()


def main():
    args, _ = parse_args()
    IMAGES_DIR_NAME = "images"

    for input_dir in args.input:
        input_dir = Path(input_dir)
        input_image_dir = Path(input_dir, IMAGES_DIR_NAME)
        input_dir_name = input_dir.name

        sequence = sorted([f for f in input_image_dir.iterdir() if f.is_file()])
        sequence_size = len(sequence)

        subset_sizes = []
        subset_sizes.extend(
            [round(float(fraction) * sequence_size) for fraction in args.fractions]
        )
        subset_sizes.extend(args.ns)

        for method in args.methods:
            for subset_size in tqdm(subset_sizes):

                if method == Method.FIRST_LAST_INTERPOLATE.value:
                    indices_picked = np.round(
                        np.linspace(
                            0, sequence_size - 1, num=subset_size, retstep=False
                        )
                    ).astype(int)
                else:  # Method.DISPERSE.value
                    step = 1 / (subset_size + 1)  # n = 3
                    fracs = np.linspace(step, 1 - step, subset_size)  # 0.25, 0.5, 0.75
                    indices_picked = np.round(fracs * (sequence_size - 1)).astype(int)

                sequence_subset = operator.itemgetter(*(indices_picked.tolist()))(
                    sequence
                )
                subset_dir_name = Path(args.output, f"{input_dir_name}_n_{subset_size}")
                subset_dir_name.mkdir(parents=True, exist_ok=True)

                # Copy colmap and transforms
                copytree(
                    input_dir,
                    subset_dir_name,
                    ignore=ignore_patterns(IMAGES_DIR_NAME),
                    dirs_exist_ok=True,
                )

                for image in sequence_subset:
                    images_dir = Path(subset_dir_name, IMAGES_DIR_NAME)
                    images_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(image, images_dir)
                    print("Saved to:", str(Path(images_dir, image.name)))


if __name__ == "__main__":
    main()
