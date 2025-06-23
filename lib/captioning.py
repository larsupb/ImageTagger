import os

import config
from lib.image_dataset import INSTANCE as DATASET
from lib.tagging.florence_tagger import generate_florence_caption
from lib.tagging.joycaption import generate_joycaption_caption
from lib.tagging.joytag.joytag import generate_joytag_caption
from lib.tagging.qwen2vl_tagger import generate_qwen2vl_caption
from lib.tagging.sbert_tagger_keywords import generate_multi_sbert
from lib.tagging.wd14_tagger import generate_wd14_caption

TAGGERS = ['joytag', 'joycaption', 'wd14', 'florence', 'qwen2-vl', 'multi_sbert', 'combo']


def generate_caption(current_index, option, state_dict):
    if not DATASET.initialized:
        return
    path = DATASET.media_paths[current_index]
    caption = ""
    if os.path.exists(path):
        if option == 'joytag':
            caption = generate_joytag_caption(path)
        elif option == 'joycaption':
            caption = generate_joycaption_caption(path, config.tagger_instruction(state_dict))
        elif option == 'wd14':
            caption = generate_wd14_caption(path)
        elif option == 'florence':
            caption = generate_florence_caption(path, prompt=config.florence_settings(state_dict)["prompt"])
        elif option == 'qwen2-vl':
            caption = generate_qwen2vl_caption(path)
        elif option == 'multi_sbert':
            caption = generate_multi_sbert(path,
                                           config.sbert_taggers(state_dict).sbert_taggers(),
                                           config.sbert_taggers(state_dict).sbert_threshold())
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


def save_caption(index, caption_text, file_name=None):
    if caption_text is not None:
        if file_name is not None:
            index = DATASET.find_index(file_name)
        DATASET.save_caption(index, caption_text)


