import gc
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from nerfstudio.scripts.eval import ComputePSNR


def main():
    model_dirs = Path("models").glob("*indices*")
    for model_dir in tqdm(model_dirs):
        yaml_config_path = model_dir / "config.yml"
        models = list(model_dir.glob("*30000*.ckpt"))
        for model in tqdm(models):
            compute_config = ComputePSNR(load_config=yaml_config_path, load_ckpt=model)
            compute_config.main()
            del compute_config
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(3)


if __name__ == "__main__":
    main()
