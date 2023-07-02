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
    parser.add_argument("--suffix", type=str, required=True)
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
    "psnr": "PSNR",
    "ssim": "SSIM",
    "lpips": "LPIPS",
}


def main():
    args = parse_args()

    # Generating unseen angles with +10 deg angless
    picked_scene_angle = 10
    test_angle_increment = 2.5
    experiment_name = f"img_size_scene_angle_{picked_scene_angle}_{args.suffix}"

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
        for scene_angle, metrics_small in metrics_big.items():
            if scene_angle != picked_scene_angle:
                continue
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
        "Maksimalni viđeni kut gledišta": [],
        "horizontalni kut gledišta": [],
        PSNR_KEY: [],
        SSIM_KEY: [],
        LPIPS_KEY: [],
    }

    for v in flattened:
        latex_dict["Maksimalni viđeni kut gledišta"].append(v["scene_angle"])
        latex_dict["horizontalni kut gledišta"].append(v["angle"])
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
        lines = df.pivot(index="angle", columns="n", values=metric).plot.line()
        degree_symbol = "\u00b0"  # Degree symbol
        print(lines.fmt_ydata)
        legend_labels = [f"{line.get_label()} {degree_symbol}" for n_unique in df["n"].unique()]

        plt.legend(title="Maksimalni viđeni kut gledišta", labels=legend_labels)
        plt.title(f"{metric_name} za neviđene kuteve gledišta")
        plt.ylabel(metric_name)
        plt.xlabel("Horizontalni kut gledišta (°)")
        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(format_degrees))
        print("Saving file:", image_filename)
        plt.savefig(image_filename)


if __name__ == "__main__":
    main()
