import re
import gradio as gr
import pandas as pd

from typing import List
from PIL import Image
from ui_symbols import process_symbol
from lib.image_dataset import INSTANCE as DATASET, load_media


def create_tag_cloud(sort_by: str):
    tags = dict()
    for i in range(0, DATASET.size()):
        for tag in DATASET.read_tags_at(i):
            if tag not in tags:
                tags[tag] = 1
            else:
                tags[tag] += 1

    items = tags.items()
    if sort_by == "Name":
        items = sorted(items, key=lambda x: x[0])
    else:
        items = sorted(items, key=lambda x: x[1], reverse=True)

    items_ = [(f'{key} ({value})', key) for key, value in items]
    return gr.CheckboxGroup(label="Tag cloud", choices=items_, value=[])


def remove_selected_tags(remove_tags: List, sort_by: str):
    for t in remove_tags:
        for i in range(0, DATASET.size()):
            tags = DATASET.read_tags_at(i)

            # Recreate caption_text by filtering the tags list
            caption_text = ", ".join([x for x in tags if x != t])

            # Save the new caption_text
            DATASET.save_caption(i, caption_text)
    return create_tag_cloud(sort_by)


body_parts = [
    # Head and Neck
    "head", "skull", "forehead", "temples", "face", "cheeks", "chin", "jaw",
    "ears", "inner ear", "outer ear", "earlobe", "nose", "nostrils", "bridge of nose",
    "mouth", "lips", "parted lips", "teeth", "upper teeth", "tongue", "palate", "throat", "neck", "nape",
    "hair", "scalp", "eyes", "eyebrows", "eyelashes", "eyelids",
    # Torso
    "chest", "ribcage", "sternum", "abdomen", "stomach", "waist", "hips", "pelvis",
    "spine", "shoulders", "collarbone", "back", "lower back", "upper back",
    "navel", "belly button",
    "large breasts", "small breasts", "medium breasts", "breasts", "breasts apart",
    # Upper Limbs
    "arm", "arm pits", "upper arm", "elbow", "forearm", "wrist", "hand", "palm", "fingers",
    "thumb", "index finger", "middle finger", "ring finger", "little finger",
    "knuckles", "nails",
    # Lower Limbs
    "leg", "thigh", "knee", "shin", "calf", "ankle", "heel", "foot", "toes",
    "big toe", "little toe", "arch", "sole", "instep",
    # Reproductive System
    "genitals", "penis", "testicles", "scrotum", "vulva", "vagina", "ovaries", "pussy",
    "uterus", "fallopian tubes", "clitoris", "cervix",
    # Miscellaneous
    "skin", "pores", "sweat glands", "sebaceous glands", "cartilage",
    "joints", "ligaments", "tendons", "bone marrow", "orange hair"
]

bad_elements = [
    "mole", "tattoo", "eyes", "nails", "dark-skinned"
]


def cleanup_tags(sort_by: str):
    for i in range(0, DATASET.size()):
        tags = DATASET.read_tags_at(i)
        tags = list(dict.fromkeys(tags))

        # remove tag if it is a body part
        tags = [tag for tag in tags if tag not in body_parts]
        # Remove tags that contain any word from bad_elements
        tags = [tag for tag in tags if not any(bad_ele in tag for bad_ele in bad_elements)]

        caption_text = ", ".join(tags)
        DATASET.save_caption(i, caption_text)
    return create_tag_cloud(sort_by)


def replace_underscores(sort_by: str):
    for i in range(0, DATASET.size()):
        caption_text = DATASET.read_caption_at(i)
        caption_text = caption_text.replace("_", " ")
        DATASET.save_caption(i, caption_text)

    return create_tag_cloud(sort_by)


def append_tag(tag: str, sort_by: str):
    for i in range(0, DATASET.size()):
        caption_text = DATASET.read_caption_at(i)
        if tag not in caption_text:
            caption_text += ", " + tag
            DATASET.save_caption(i, caption_text)
    return create_tag_cloud(sort_by)


def prepend_tag(tag: str, sort_by: str):
    for i in range(0, DATASET.size()):
        caption_text = DATASET.read_caption_at(i)
        if tag not in caption_text:
            caption_text = tag + ", " + caption_text
            DATASET.save_caption(i, caption_text)
    return create_tag_cloud(sort_by)


def refresh_thumbnails(tags: list, inverse: bool):
    if not DATASET.initialized:
        return
    image_controls = []

    def check_relevance(index, img_path, mask_path, caption):
        image_tags = DATASET.read_tags_at(index)
        # if any element of tags is in image_tags, create a gr.Image object
        if not inverse and any(tag in image_tags for tag in tags):
            # read image as PIL.Image and add it to the list
            image_controls.append(load_media(img_path))
        # if inverse mode and none of the elements of tags is in image_tags, create a gr.Image object
        if inverse and all(tag not in image_tags for tag in tags):
            image_controls.append(load_media(img_path))

    DATASET.scan(check_relevance)
    return image_controls


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


def tab_captions() -> gr.Gallery:
    with gr.Tab("Captions"):
        with gr.Accordion("Tag cloud"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        button_create_tag_cloud = gr.Button("Refresh cloud")
                        checkbox_inverse_filter = gr.Checkbox(label="inverse (having not)", value=False)
                        sort_by = gr.Radio(label="Sort by", choices=["Name", "Count"], value="Name")
                    checkbox_tag_cloud = gr.CheckboxGroup(label="Tag cloud", )
                with gr.Column():
                    button_remove_tags = gr.Button("Remove selected tags")
                    button_cleanup_tags = gr.Button("Cleanup tags")
                    button_replace_underscores = gr.Button("Replace underscores")
                    with gr.Row():
                        textbox_add_tag = gr.Textbox(placeholder="Enter a new tag", )
                        with gr.Row():
                            button_prepend_tag = gr.Button(value="Prepend tag")
                            button_append_tag = gr.Button(value="Append tag")
                    thumb_view = gr.Gallery(allow_preview=True, preview=False, columns=2)

            button_create_tag_cloud.click(create_tag_cloud, inputs=sort_by, outputs=checkbox_tag_cloud)
            button_remove_tags.click(remove_selected_tags, inputs=[checkbox_tag_cloud, sort_by], outputs=checkbox_tag_cloud)
            button_cleanup_tags.click(cleanup_tags, inputs=sort_by, outputs=checkbox_tag_cloud)
            button_replace_underscores.click(replace_underscores, inputs=sort_by, outputs=checkbox_tag_cloud)
            button_prepend_tag.click(prepend_tag, inputs=[textbox_add_tag, sort_by], outputs=checkbox_tag_cloud)
            button_append_tag.click(append_tag, inputs=[textbox_add_tag, sort_by], outputs=checkbox_tag_cloud)

            checkbox_tag_cloud.change(refresh_thumbnails, inputs=[checkbox_tag_cloud, checkbox_inverse_filter],
                                      outputs=thumb_view)

        with gr.Accordion("Search and replace", open=False):
            with gr.Row():
                textbox_search_for = gr.Textbox(label="Search", placeholder="Search term")
                textbox_replace_with = gr.Textbox(label="Replace", placeholder="Replace with")
                button_batch_search = gr.Button(value=process_symbol + "Search and replace (preview)")
                button_batch_replace = gr.Button(value=process_symbol + "Replace (save)")

            with gr.Row():
                df_search_results = gr.DataFrame(headers=["Index", "Caption", "Caption modified"])

            button_batch_search.click(caption_search, inputs=[textbox_search_for, textbox_replace_with],
                                      outputs=df_search_results)
            button_batch_replace.click(caption_search_and_replace, inputs=[textbox_search_for, textbox_replace_with])

            # button_batch_replace = gr.Button(value=process_symbol + " Replace", elem_id="button_replace")
            # inputs = [textbox_editing_search_for, textbox_editing_replace_with]

    return thumb_view
