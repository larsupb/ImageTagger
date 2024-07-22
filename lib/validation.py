import os

import pandas as pd
from PIL import Image
from lib.image_aspects import ImageAspects
from lib.image_dataset import INSTANCE as DATASET


def validate_dataset(aspect_ratios):
    if aspect_ratios is not None and len(aspect_ratios) > 0:
        aspects = [ImageAspects.from_code(a) for a in aspect_ratios]
    else:
        aspects = [ImageAspects.SQUARE,
                   ImageAspects.POSTCARD,
                   ImageAspects.POSTCARD_INVERSE]

    def meets_format(d: tuple, epsilon=0.01):
        ratios_allowed = [a.value for a in aspects]
        ratio = d[0]/d[1]
        for da in ratios_allowed:
            if da - da * epsilon < ratio < da + da * epsilon:
                return True
        print(f"Missing format {d}: {ratio}")
        return False

    def validate(index, img_path, mask_path, caption):
        img_validation = {'Image': '', 'Mask': '', 'Caption': ''}
        if img_path is None or not os.path.exists(img_path):
            img_validation['Image'] = f'No image found'
        elif mask_path is None or not os.path.exists(mask_path):
            img_validation['Mask'] = f'No mask found'
        else:
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            if img.size != mask.size:
                img_validation['Image'] = f'Image dims do not fit to mask dims!'

        if img_path is not None and os.path.exists(img_path):
            img = Image.open(img_path)
            if not meets_format(img.size):
                img_validation['Image'] = f'Not in the allowed aspect ratios!'
        if caption is None or len(caption) == 0:
            img_validation['Caption'] = f'Caption is empty!'

        validation_results.append(img_validation)

    if DATASET.empty():
        return None

    validation_results = []
    # Scan the dataset and validate each image
    DATASET.scan(validate)
    return pd.DataFrame(data=validation_results, columns=['Image', 'Mask', 'Caption']).reset_index()
