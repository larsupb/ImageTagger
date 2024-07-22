import re
from typing import List

import gradio as gr
import pandas as pd

from lib.image_dataset import INSTANCE as DATASET


def create_tag_cloud():
    tags = set()
    for i in range(0, DATASET.size()):
        tags.update(DATASET.read_tags_at(i))
    return gr.update(choices=(sorted(list(tags))))


def remove_selected_tags(remove_tags: List):
    for t in remove_tags:
        for i in range(0, DATASET.size()):
            tags = DATASET.read_tags_at(i)

            # Recreate caption_text by filtering the tags list
            caption_text = ", ".join([x for x in tags if x != t])

            # Save the new caption_text
            DATASET.save_caption(i, caption_text)
    return create_tag_cloud()


def remove_duplicates():
    for i in range(0, DATASET.size()):
        tags = DATASET.read_tags_at(i)
        tags = list(dict.fromkeys(tags))
        caption_text = ", ".join(tags)
        DATASET.save_caption(i, caption_text)
    return create_tag_cloud()


def replace_underscores():
    for i in range(0, DATASET.size()):
        caption_text = DATASET.read_caption_at(i)
        caption_text = caption_text.replace("_", " ")
        DATASET.save_caption(i, caption_text)

        remove_duplicates()
    return create_tag_cloud()


def append_tag(tag: str):
    for i in range(0, DATASET.size()):
        caption_text = DATASET.read_caption_at(i)
        if tag not in caption_text:
            caption_text += ", " + tag
            DATASET.save_caption(i, caption_text)
    return create_tag_cloud()


def prepend_tag(tag: str):
    for i in range(0, DATASET.size()):
        caption_text = DATASET.read_caption_at(i)
        if tag not in caption_text:
            caption_text = tag + ", " + caption_text
            DATASET.save_caption(i, caption_text)
    return create_tag_cloud()


def caption_search(search_for, replace_with):
    if not DATASET.initialized:
        return
    # Compile a case-insensitive regular expression for the search term
    pattern = re.compile(re.escape(search_for), re.IGNORECASE)
    results = []
    for i in range(0, DATASET.size()):
        caption_text = DATASET.read_caption_at(i)
        modified_text = caption_text
        matches = list(pattern.finditer(caption_text))
        for match in reversed(matches):
            start, end = match.start(), match.end()
            # Replace matched text with the same case as the replacement text
            modified_text = modified_text[:start] + re.sub(pattern, replace_with,
                                                           modified_text[start:end]) + modified_text[end:]
        if matches:
            results.append([i, caption_text, modified_text])

    new_df = pd.DataFrame(results, columns=["Index", "Caption", "Caption modified"])
    return gr.update(value=new_df)


def caption_search_and_replace(search_for, replace_with):
    if not DATASET.initialized:
        return
    # Compile a case-insensitive regular expression for the search term
    pattern = re.compile(re.escape(search_for), re.IGNORECASE)
    for i in range(0, DATASET.size()):
        caption_text = DATASET.read_caption_at(i)
        # Replace all occurrences of the search term with the replacement term
        # while preserving the case of the rest of the caption
        caption_text_mod = pattern.sub(replace_with, caption_text)
        DATASET.save_caption(i, caption_text_mod)
