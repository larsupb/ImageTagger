import os

import numpy as np

from lib.image_dataset import ImageDataSet

if __name__ == "__main__":
    print("Enter the path to the root directory with concept folders:")
    path = input()

    # scan each subdirectory for images
    registry = dict()
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        if not os.path.isdir(dir_path):
            continue

        print(f"Scanning {dir}")
        dataset = ImageDataSet()
        dataset.load(path=dir_path, subdirectories=True, load_thumbnails=False)

        print(f"Dataset size: {len(dataset)}")
        registry[dir] = len(dataset)

    print("Registry:")
    for k, v in registry.items():
        print(f"{k}: {v}")

    # Calculate a num_repeats value for each concept so that each concept has the same number of images
    num_avg_repeats = 20
    max_repeats = 26
    min_repeats = 14
    median = np.median(list(registry.values()))
    for concept in registry.keys():
        registry[concept] = median / registry[concept]
        registry[concept] *= num_avg_repeats
        registry[concept] = int(max(min_repeats, min(max_repeats, registry[concept])))
    print("Registry after scaling:")
    for k, v in registry.items():
        print(f"{k}: {v}")

    # -------------------------------------------------------------
    # Create a config file
    concept_class = "woman"
    content = """
    [[datasets]]  
    """
    for concept, num_repeats in registry.items():
        content += f"""
        [[datasets.subsets]]
        image_dir = '/{concept}'                
        num_repeats = {num_repeats}
        keep_tokens = 2
        caption_extension = ".txt"
        class_tokens = '{concept_class}'
        """

    # Write the config file
    with open(os.path.join(path, "dataset_config.toml"), "w") as f:
        f.write(content)




