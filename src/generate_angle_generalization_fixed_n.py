import argparse
import gc
import json
import re
import time
from calendar import c
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import tabulate
import torch
import yaml
from arrow import get
from matplotlib import pyplot as plt
from regex import P
from tqdm import tqdm

from nerfstudio.scripts.my_utils import (
    get_angle_from_dir_name,
    get_n_from_dir_name,
    metrics_agg_by_angle,
)


def format_degrees(value, pos=None):
    return f"{value:.2f} °"


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--suffix", type=str, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--metric_json_name", type=Path, default="step-000030000_metrics_test.json")
    args = parser.parse_args()
    return args


metrics_fullname = {
    "psnr": "Peak Signal to Noise Ratio",
    "ssim": "Structural Similarity Index",
    "lpips": "Learned Perceptual Image Patch Similarity",
}

metrics_capitalized = {
    "psnr": r"PSNR $\uparrow$",
    "ssim": r"SSIM $\uparrow$",
    "lpips": r"LPIPS $\downarrow$",
}


def main():
    args = parse_args()

    # Generating unseen angles with +10 deg angless
    test_angle_increment = 2.5
    picked_n = 121

    suffix = ""
    is_test = "test" in str(args.metric_json_name)
    is_bounded = "bounded" in str(args.metric_json_name)
    suffix += "_test" if is_test else "_val"
    suffix += "_bounded" if is_bounded else "_unbounded"

    if "card_mask" in str(args.checkpoints[0]):
        suffix += "card_mask"
    if "hand_mask" in str(args.checkpoints[0]):
        suffix += "hand_mask"

    experiment_name = f"angle_generalization_n_{picked_n}_{suffix}"

    # {
    #     10: # scene angle
    #       25: # offset angle
    #           64: { # n
    #               "psnr": 3
    #               "ssim": 3
    #               "lpips": 3
    #           }
    # }

    result: dict[str, dict[str, float]] = {}
    for checkpoint in tqdm(args.checkpoints):
        checkpoint: Path
        model_dir = checkpoint.parent
        # yaml_config_path = model_dir / "config.yml"
        # models = list(model_dir.glob("*.ckpt"))
        # final_step_models = [model for model in models if ("30000" in str(model))]

        # if len(final_step_models) == 0:
        #     print("warning: no final step model found for", model_dir)
        #     continue
        scene_angle = get_angle_from_dir_name(model_dir)
        if scene_angle not in result:
            result[scene_angle] = {}
        num_images = get_n_from_dir_name(model_dir)

        metric_json = model_dir / args.metric_json_name

        test_angles = [scene_angle + test_angle_increment]
        while test_angles[-1] < 60:
            test_angles.append(test_angles[-1] + test_angle_increment)
        # test_angles = [scene_angle + (i * test_angle_increment) for i in range(1, num_test_images + 1)]
        # test_angles = [35, 40, 45]

        metrics = json.load(open(str(metric_json)))
        metrics_angle = metrics_agg_by_angle(metrics["images"])

        for test_angle in test_angles:
            if num_images not in result:
                result[num_images] = {}
            if scene_angle not in result[num_images]:
                result[num_images][scene_angle] = {}
            if test_angle in result[num_images][scene_angle]:
                print(
                    f"WARNING:, value at {test_angle}:{num_images} alreay exists: {result[num_images][scene_angle][test_angle]}"
                )
            result[num_images][scene_angle][test_angle] = metrics_angle[test_angle]
    print(json.dumps(result, indent=4))
    flattened = []

    for n, metrics_big in result.items():
        if n != picked_n:
            continue
        for scene_angle, metrics_small in metrics_big.items():
            for angle, metric_dict in metrics_small.items():
                # for scene_angle, scene_angle_metrics in result.items():
                #     for angle, scene_angle_offset_metrics in scene_angle_metrics.items():
                #         for n, metric_dict in scene_angle_offset_metrics.items():
                flattened.append(
                    {
                        "scene_angle": scene_angle,
                        "angle": angle,
                        "n": n,
                        "psnr": metric_dict["psnr"],
                        "ssim": metric_dict["ssim"],
                        "lpips": metric_dict["lpips"],
                    }
                )

    PSNR_KEY = r"PSNR \uparrow"
    SSIM_KEY = r"SSIM \uparrow"
    LPIPS_KEY = r"LPIPS \downarrow"
    latex_dict = {
        "Maksimalni viđeni kut smjera očišta": [],
        "horizontalni kut smjera očišta": [],
        PSNR_KEY: [],
        SSIM_KEY: [],
        LPIPS_KEY: [],
    }

    for v in flattened:
        latex_dict["Maksimalni viđeni kut smjera očišta"].append(v["scene_angle"])
        latex_dict["horizontalni kut smjera očišta"].append(v["angle"])
        latex_dict[PSNR_KEY].append(f"{v['psnr']:.2f}")
        latex_dict[SSIM_KEY].append(f"{v['ssim']:.2f}")
        latex_dict[LPIPS_KEY].append(f"{v['lpips']:.2f}")
    latex_table = tabulate.tabulate(
        latex_dict,
        tablefmt="latex_raw",
        headers="keys",
    )

    latex_table_txt = f"{experiment_name}.txt"
    print("Saving file", latex_table_txt)
    Path(latex_table_txt).write_text(latex_table)

    for metric in ["psnr", "ssim", "lpips"]:
        metric_name = metrics_capitalized[metric]
        image_filename = f"{experiment_name}_{metric}.png"
        drop_metrics = [m for m in ["psnr", "ssim", "lpips"] if m != metric]

        # for scene_angle in result.keys():

        df = pd.DataFrame(flattened)
        df.drop(drop_metrics, axis=1, inplace=True)
        df.pivot(index="angle", columns="scene_angle", values=metric).plot.line()

        degree_symbol = "\u00b0"  # Degree symbol
        legend_labels = [f"{n_unique}{degree_symbol}" for n_unique in df["scene_angle"].unique()]

        plt.legend(title="Maksimalni viđeni kut smjera očišta", labels=legend_labels)
        plt.title(f"{metric_name} za neviđene kuteve smjera očišta")
        plt.ylabel(metric_name)
        plt.xlabel("Horizontalni kut smjera očišta (°)")
        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(format_degrees))
        print("Saving file:", image_filename)
        plt.savefig(image_filename)


if __name__ == "__main__":
    main()
