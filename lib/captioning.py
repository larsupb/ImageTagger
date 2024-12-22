import os

import config
from lib.image_dataset import INSTANCE as DATASET
from lib.tagging import generate_rag_caption
from lib.tagging.florence_tagger import generate_florence_caption
from lib.tagging.joycaption import generate_joycaption_caption
from lib.tagging.joytag.joytag import generate_joytag_caption
from lib.tagging.qwen2vl_tagger import generate_qwen2vl_caption
from lib.tagging.sbert_tagger_keywords import generate_multi_sbert
from lib.tagging.wd14_tagger import generate_wd14_caption

TAGGERS = ['joytag', 'joycaption', 'wd14', 'florence', 'qwen2-vl', 'rag', 'multi_sbert']


def generate_caption(current_index, option, state_dict):
    if not DATASET.initialized:
        return
    path = DATASET.image_paths[current_index]
    caption = ""
    if os.path.exists(path):
        if option == 'joytag':
            caption = generate_joytag_caption(path)
        elif option == 'joycaption':
            caption = generate_joycaption_caption(path, config.tagger_instruction(state_dict))
        elif option == 'wd14':
            caption = generate_wd14_caption(path)
        elif option == 'florence':
            caption = generate_florence_caption(path)
        elif option == 'qwen2-vl':
            caption = generate_qwen2vl_caption(path)
        elif option == 'multi_sbert':
            caption = generate_multi_sbert(path,
                                           config.sbert_taggers(state_dict).sbert_taggers(),
                                           config.sbert_taggers(state_dict).sbert_threshold())
        else:
            caption = generate_rag_caption(path)
    return caption


def save_caption(index, caption_text, file_name=None):
    if caption_text is not None:
        if file_name is not None:
            index = DATASET.find_index(file_name)
        DATASET.save_caption(index, caption_text)


