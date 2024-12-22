import os

import numpy as np
from PIL import Image

import config


def ask_mask_from_model(image, model):
    import rembg
    os.environ['U2NET_HOME'] = os.path.join(CONFIG.models_dir(), 'rembg')
    os.makedirs(os.environ['U2NET_HOME'], exist_ok=True)

    img_out = rembg.remove(image, post_process_mask=True, session=rembg.new_session(model), only_mask=True)

    img_out = img_out.convert('RGBA')

    data = np.array(img_out)  # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability

    # Replace white with red... (leaves alpha values alone...)
    white_areas = (red == 255) & (blue == 255) & (green == 255)
    data[..., :-1][white_areas.T] = (255, 0, 0)  # Transpose back needed

    # Replace black with transparent areas (leaves alpha values alone...)
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    data[...][black_areas.T] = (0, 0, 0, 0)  # Transpose back needed

    img_out = Image.fromarray(data)

    return img_out


def remove_background(image, model, state_dict: dict):
    import rembg
    os.environ['U2NET_HOME'] = os.path.join(config.models_dir(state_dict), 'rembg')
    os.makedirs(os.environ['U2NET_HOME'], exist_ok=True)

    img_out = rembg.remove(image, post_process_mask=True, session=rembg.new_session(model), only_mask=False)
    img_out = img_out.convert('RGBA')
    return img_out
