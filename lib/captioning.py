import os

import config
from lib.image_dataset import ImageDataSet
from lib.tagging.florence_tagger import generate_florence_caption
from lib.tagging.joytag.joytag import generate_joytag_caption
from lib.tagging.openai_tagger import generate_openai_caption
from lib.tagging.qwen2vl_tagger import generate_qwen2vl_caption
from lib.tagging.wd14_tagger import generate_wd14_caption

TAGGERS = ['joytag', 'wd14', 'florence', 'qwen2-vl', 'openai', 'combo']


def _get_dataset(state_dict: dict) -> ImageDataSet | None:
    """Helper to extract dataset from state dict."""
    if state_dict is None:
        return None
    return state_dict.get('dataset')


def generate_caption(current_index, option, state_dict):
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.initialized:
        return
    path = dataset.media_paths[current_index]
    caption = ""
    if os.path.exists(path):
        if option == 'joytag':
            caption = generate_joytag_caption(path)
        elif option == 'wd14':
            caption = generate_wd14_caption(path)
        elif option == 'florence':
            caption = generate_florence_caption(path, prompt=config.florence_settings(state_dict)["prompt"])
        elif option == 'qwen2-vl':
            caption = generate_qwen2vl_caption(path)
        elif option == 'openai':
            caption = generate_openai_caption(path, config.openai_settings(state_dict))
        elif option == 'combo':
            caption = generate_combo_caption(current_index, state_dict)
    return caption


def generate_combo_caption(current_index, state_dict):
    """
    Generate a caption using a combination of taggers
    """
    caption = []
    taggers = config.combo_taggers(state_dict)
    for t in taggers:
        caption.append(generate_caption(current_index, t, state_dict))
    return ', '.join(caption)


def save_caption(index, caption_text, file_name=None, dataset: ImageDataSet = None):
    """
    Save caption text for an image.

    :param index: Image index, or -1 if using file_name
    :param caption_text: Caption text to save
    :param file_name: Optional file name to look up index
    :param dataset: ImageDataSet instance (required)
    """
    if caption_text is None or dataset is None or not dataset.initialized:
        return
    if file_name is not None:
        index = dataset.find_index(file_name)
    dataset.save_caption(index, caption_text)
