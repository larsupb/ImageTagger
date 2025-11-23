import os
import shutil
import tempfile

import PIL
import gradio as gr
from PIL import Image

import config
from lib.image_dataset import ImageDataSet
from lib.media_item import is_video


def init_dataset(path: str, masks_path: str, filter_missing_captions: bool, subdirectories: bool, load_gallery: bool, state_dict: dict):
    """
    Initialize the dataset and load the first image.
    Creates a new ImageDataSet instance and stores it in state_dict.

    :param path: Path to the image dataset
    :param masks_path: Path to the masks dataset
    :param filter_missing_captions: If True, images without captions will be filtered out
    :param subdirectories: If True, images will be loaded from subdirectories
    :param load_gallery: If True, the gallery will be loaded
    :param state_dict: State dictionary - will contain 'dataset' key after init
    :return: gallery, slider, loader_data (-- skip index), state_dict
    """
    # Create new dataset instance for this session
    dataset = ImageDataSet()
    dataset.prune_orphaned_captions(path, subdirectories)
    dataset.load(path, masks_path, filter_missing_captions, config.ignore_list(state_dict), subdirectories, load_gallery)

    # Store dataset in state for per-session access
    state_dict['dataset'] = dataset

    if len(dataset) == 0:
        return gr.skip()

    loader_data = _navigate_forward_internal(-1, None, dataset)
    images_total = len(dataset) - 1
    slider_new = gr.Slider(value=0, minimum=0, maximum=images_total, label="Image index", step=1, interactive=True)

    gallery = gr.Gallery(value=dataset.get_all_thumbnails() if load_gallery else [], allow_preview=False, preview=False, columns=8, type="pil")
    return gallery, slider_new, *list(loader_data[1:]), state_dict


def _get_dataset(state_dict: dict) -> ImageDataSet | None:
    """Helper to extract dataset from state dict."""
    if state_dict is None:
        return None
    return state_dict.get('dataset')


def load_index(index, state_dict: dict) -> dict:
    """
    Load the image at the given index
    :param index: Image index to load
    :param state_dict: State dictionary containing 'dataset' key
    :return: dictionary contains index, dataset_size, path, byte_size_str, dimensions, caption_text, img_edit, img_mask
    """
    dataset = _get_dataset(state_dict)
    if index < 0 or dataset is None or not dataset.is_initialized:
        return {}

    item = dataset.get_item(index)
    if item is None:
        return {}

    path = item.media_path
    mask_path = item.mask_path

    img_edit = dict()
    img_edit["composite"] = None
    img_edit["layers"] = None

    img_byte_size = 0
    dimensions = ""
    if os.path.exists(path):
        img_byte_size = os.path.getsize(path)

        # Determine dimensions
        if item.is_video:
            dimensions = "Video"
        else:
            media = PIL.Image.open(path)
            dimensions = f'{media.size[0]} x {media.size[1]}'
            img_edit["background"] = media
    byte_size_str = f'{img_byte_size / 1024:.2f} kB'

    img_mask = None
    if mask_path and os.path.exists(mask_path):
        img_mask = Image.open(mask_path)
    caption_text = dataset.read_caption(index)

    # The displayed path is the original path, not the temp one
    path_text = path

    if item.is_video:
        video_ext = item.extension
        with tempfile.NamedTemporaryFile(dir=tempfile.gettempdir(), delete=False, suffix=video_ext) as temp_path:
            # Copy video to a temp file to avoid locking issues
            shutil.copy2(path, temp_path.name)
            path = temp_path.name
    out = {
        "index": index,
        "dataset_size": len(dataset),
        "path": path_text,
        "byte_size_str": byte_size_str,
        "dimensions": dimensions,
        "caption_text": caption_text,
        "img_edit": None if item.is_video else img_edit,
        "img_mask": None if item.is_video else img_mask,
        "video_display": path if item.is_video else None
    }
    print(out)
    return out


def _navigate_forward_internal(current_index, caption_text, dataset: ImageDataSet):
    """Internal helper for navigation during init (before state is set up)."""
    from lib.captioning import save_caption
    save_caption(current_index, caption_text, dataset=dataset)
    new_index = min(current_index + 1, len(dataset) - 1)
    if new_index < 0:
        return None
    # Create a temporary state dict for load_index
    temp_state = {'dataset': dataset}
    return to_control_group(load_index(new_index, temp_state))


def navigate_forward(current_index, caption_text, state_dict: dict):
    """
    Save the caption and load the next image
    """
    from lib.captioning import save_caption
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.is_initialized:
        return gr.skip()

    save_caption(current_index, caption_text, dataset=dataset)
    new_index = min(current_index + 1, len(dataset) - 1)
    if new_index < 0:
        return gr.skip()
    return to_control_group(load_index(new_index, state_dict))


def navigate_backward(current_index, caption_text, state_dict: dict):
    """
    Save the caption and load the previous image
    Does not return components like gallery or slider
    """
    from lib.captioning import save_caption
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.is_initialized:
        return gr.skip()

    save_caption(current_index, caption_text, dataset=dataset)
    new_index = max(current_index - 1, 0)
    if new_index < 0:
        return gr.skip()
    return to_control_group(load_index(new_index, state_dict))


def toggle_bookmark(current_index, state_dict: dict):
    """Toggle bookmark status for an image."""
    bookmark_on = "🔖"  # symbol when toggled on
    bookmark_off = "📑"  # symbol when toggled off

    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.is_initialized:
        return bookmark_off

    # toggle and get new state
    new_state = dataset.toggle_bookmark(current_index)
    # determine new symbol
    new_symbol = bookmark_on if new_state else bookmark_off

    # Change bookmark button state
    return new_symbol


def jump(new_index, caption_text, file_name, state_dict: dict):
    """Jump to a specific index."""
    from lib.captioning import save_caption
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.is_initialized:
        return gr.skip()

    save_caption(-1, caption_text, file_name, dataset=dataset)
    return to_control_group(load_index(new_index, state_dict))


def to_control_group(nav_data: dict) -> tuple:
    return (nav_data["index"],
            nav_data["dataset_size"], nav_data["path"],
            nav_data["byte_size_str"], nav_data["dimensions"], nav_data["caption_text"],
            nav_data["img_edit"], nav_data["img_mask"],
            nav_data["video_display"])
