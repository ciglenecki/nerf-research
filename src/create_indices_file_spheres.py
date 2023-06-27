import json
import random
from pathlib import Path


def main():
    directories = list(Path("data").glob("n_*"))
    min_directory_len = min([len(directory.name) for directory in directories])

    dir_groups = {}
    for directory in directories:
        prefix = directory.name[:min_directory_len]
        if prefix not in dir_groups:
            dir_groups[prefix] = []
        dir_groups[prefix].append(directory)

    split = 0.25

    for directories in dir_groups.values():
        directory = directories[0]
        store = {}
        # store_train = {}
        # store_test = {}

        img_paths = Path(directory, "images").glob("*.png")
        for img_path in img_paths:
            key = str(Path("images", img_path.name))
            is_test = "test" in img_path.name
            if is_test:
                store[key] = "test"
            else:
                store[key] = "train"
                # store_test[key] = "train"
                # store_train[key] = "train"

        # Split train to train and val
        train_imgs = [key for key, value in store.items() if value == "train"]
        num_val = int(len(train_imgs) * split)
        val_imgs = random.sample(list(train_imgs), num_val)
        for val_img in val_imgs:
            store[val_img] = "val"

        for directory in directories:
            with open(Path(directory, "indices.json"), "w") as f:
                json.dump(store, f, indent=4, sort_keys=True)

        # with open(Path(directory, "indices.json"), "w") as f:
        #     json.dump(store, f, indent=4, sort_keys=True)

        # # Save train/val split
        # with open(Path(directory, "indices_train.json"), "w") as f:
        #     json.dump(store_train, f, indent=4, sort_keys=True)

        # # Save test split
        # with open(Path(directory, "indices_test.json"), "w") as f:
        #     json.dump(store_test, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
