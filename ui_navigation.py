import os

import gradio as gr
from PIL import Image

import config
from lib.captioning import save_caption
from lib.image_dataset import INSTANCE as DATASET


def init_dataset(path: str, masks_path: str, filter_missing_captions: bool, subdirectories: bool, load_gallery: bool, state_dict: dict):
    """
    Initialize the dataset and load the first image
    :param path: Path to the image dataset
    :param masks_path: Path to the masks dataset
    :param filter_missing_captions: If True, images without captions will be filtered out
    :param subdirectories: If True, images will be loaded from subdirectories
    :param load_gallery: If True, the gallery will be loaded
    :param state_dict: State dictionary
    :return: gallery, slider, loader_data (-- skip index) [dataset_size, path, byte_size_str, dimensions, caption_text, img_edit, img_mask]
    """
    DATASET.prune(path, subdirectories)
    DATASET.load(path, masks_path, filter_missing_captions, config.ignore_list(state_dict), subdirectories, load_gallery)

    if DATASET.size() == 0:
        return gr.skip()

    loader_data = navigate_forward(-1, None)
    images_total = DATASET.size() - 1
    slider_new = gr.Slider(value=0, minimum=0, maximum=images_total, label="Image index", step=1, interactive=True)

    gallery = gr.Gallery(value=DATASET.images, allow_preview=False, preview=False, columns=8, type="pil")
    return gallery, slider_new, *list(loader_data[1:])


def load_index(index) -> dict:
    """
    Load the image at the given index
    :param index:
    :return: dictionary contains index, dataset_size, path, byte_size_str, dimensions, caption_text, img_edit, img_mask
    """
    if index < 0:
        return {}

    path = DATASET.image_paths[index]
    mask_path = DATASET.mask_paths(index)

    img_edit = dict()
    img_edit["composite"] = None
    img_edit["layers"] = None

    img_byte_size = 0
    dimensions = ""
    if os.path.exists(path):
        img = Image.open(path)
        img_byte_size = os.path.getsize(path)
        dimensions = f'{img.size[0]} x {img.size[1]}'
        img_edit["background"] = img
    byte_size_str = f'{img_byte_size / 1024:.2f} kB'

    img_mask = None
    if os.path.exists(mask_path):
        img_mask = Image.open(mask_path)
    caption_text = DATASET.read_caption_at(index)

    return {
        "index": index,
        "dataset_size": DATASET.size(),
        "path": path,
        "byte_size_str": byte_size_str,
        "dimensions": dimensions,
        "caption_text": caption_text,
        "img_edit": img_edit,
        "img_mask": img_mask
    }


def navigate_forward(current_index, caption_text):
    """
    Save the caption and load the next image
    """
    save_caption(current_index, caption_text)
    new_index = min(current_index + 1, DATASET.size() - 1)
    if new_index < 0:
        return
    return to_control_group(load_index(new_index))


def navigate_backward(current_index, caption_text):
    """
    Save the caption and load the previous image
    Does not return components like gallery or slider
    """
    save_caption(current_index, caption_text)
    new_index = max(current_index - 1, 0)
    if new_index < 0:
        return
    return to_control_group(load_index(new_index))


def jump(new_index, caption_text, file_name):
    save_caption(-1, caption_text, file_name)
    return to_control_group(load_index(new_index))


def to_control_group(nav_data: dict) -> tuple:
    return nav_data["index"], nav_data["dataset_size"], nav_data["path"], \
        nav_data["byte_size_str"], nav_data["dimensions"], nav_data["caption_text"],\
        nav_data["img_edit"], nav_data["img_mask"]