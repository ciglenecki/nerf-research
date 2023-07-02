import gc
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tabulate
import torch
import yaml
from arrow import get
from regex import P
from tqdm import tqdm

from nerfstudio.scripts.my_utils import (
    get_angle_from_dir_name,
    get_n_from_dir_name,
    metrics_agg_by_angle,
)

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
    model_dirs = Path("models").glob("*indices*nerfacto*")
    # Generating unseen angles with +10 deg angless

    num_test_images = 5
    test_angle_increment = 5

    # {
    #     15: # angle
    #         64: { # n
    #             "psnr": 3
    #             "ssim": 3
    #             "lpips": 3
    #         }
    # }

    result: dict[str, dict[str, float]] = {}
    for model_dir in tqdm(model_dirs):
        # yaml_config_path = model_dir / "config.yml"
        models = list(model_dir.glob("*.ckpt"))
        final_step_models = [model for model in models if ("30000" in str(model))]

        if len(final_step_models) == 0:
            print("warning: no final step model found for", model_dir)
            continue
        scene_angle = get_angle_from_dir_name(model_dir)
        num_images = get_n_from_dir_name(model_dir)

        metrics_jsons = list(model_dir.glob("*metrics_test*.json"))
        if len(metrics_jsons) == 0:
            print(f"warning: no metric for model {str(model_dir)}")
            continue

        test_angles = [scene_angle + (i * test_angle_increment) for i in range(1, num_test_images + 1)]
        # test_angles = [35, 40, 45]

        for metric_json in metrics_jsons:
            # is_test = "test" in str(metric_json)
            metrics = json.load(open(str(metric_json)))
            metrics_angle = metrics_agg_by_angle(metrics["images"])

            for test_angle in test_angles:
                if test_angle not in result:
                    result[test_angle] = {}
                if num_images not in result[test_angle]:
                    result[test_angle][num_images] = {}
                if result[test_angle][num_images]:
                    print(
                        f"WARNING:, value at {test_angle}:{num_images} alreay exists: {result[test_angle][num_images]}"
                    )
                result[test_angle][num_images] = metrics_angle[test_angle]

        flattened = []

        for angle, angle_n_metric in result.items():
            for n, metric_dict in angle_n_metric.items():
                flattened.append(
                    {
                        "angle": angle,
                        "n": n,
                        "psnr": metric_dict["psnr"],
                        "ssim": metric_dict["ssim"],
                        "lpips": metric_dict["lpips"],
                    }
                )

        latex_dict = {
            "horizontalni kut gledišta": [],
            "br. slika u skupu za učenje": [],
            "PSNR": [],
            "SSIM": [],
            "LPIPS": [],
        }

        for v in flattened:
            latex_dict["horizontalni kut gledišta"].append(v["angle"])
            latex_dict["br. slika u skupu za učenje"].append(v["n"])
            latex_dict["PSNR"].append(f"{v['psnr']:.2f}")
            latex_dict["SSIM"].append(f"{v['ssim']:.2f}")
            latex_dict["LPIPS"].append(f"{v['lpips']:.2f}")
        latex_table = tabulate.tabulate(
            flattened,
            tablefmt="latex_raw",
        )
        print(latex_table)

        for metric in ["psnr", "ssim", "lpips"]:
            metric_name = metrics_capitalized[metric]
            image_filename = f"angle_generalization_{metric}_scene_{angle:.2f}.png"
            drop_metrics = [m for m in ["psnr", "ssim", "lpips"] if m != metric]
            df = pd.DataFrame(flattened)
            df.drop(drop_metrics, axis=1, inplace=True)
            print(df)

            degree_symbol = "\u00b0"
            legend_labels = [f"{a}{degree_symbol}" for a in df["scene_angle"].unique()]

            df.pivot(index="angle", columns="n", values=metric).plot.line()
            plt.legend(title="br. slika u skupu za učenje", labels=legend_labels)
            plt.title(f"{metric_name} za neviđene kuteve gledišta")
            plt.ylabel(metric_name)
            plt.savefig(image_filename)


if __name__ == "__main__":
    main()
