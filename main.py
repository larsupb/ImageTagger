import os

import gradio as gr
import numpy as np
from PIL import Image
from gradio.components.image_editor import EditorValue

from lib.captioning import create_tag_cloud, remove_selected_tags, caption_search, caption_search_and_replace, \
    prepend_tag, append_tag
from lib.common_gui import get_folder_path
from lib.image_aspects import ImageAspects
from lib.image_dataset import INSTANCE as DATASET
from config import CONFIG

from lib.masking import ask_mask_from_model, remove_background
from lib.tagging import generate_rag_caption
from lib.tagging.joytag.joytag import generate_joytag_caption
from lib.tagging.sbert_tagger_keywords import generate_multi_sbert
from lib.tagging.tagging import generate_florence_caption
from lib.tagging.wd14_tagger import generate_wd14_caption
from lib.upscaling import upscale_image, Upscalers
from lib.validation import validate_dataset

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'  # 📄
delete_symbol = u'\u232B'
process_symbol = u'\u21A9'
backward_symbol = u'\u25C0'
forward_symbol = u'\u25B6'

selected_index = -1


def init_dataset(path: str, masks_path: str, filter_missing_captions, autogenerate_mask):
    DATASET.load(path, masks_path, filter_missing_captions)

    new_index, path, caption, size, dimensions, image, image_mask, _ = navigate_forward(-1, None, autogenerate_mask)
    images_total = DATASET.size() - 1
    slide_new = gr.Slider(value=0, minimum=0, maximum=images_total, label="Image index", step=1, interactive=True)
    return slide_new, images_total, path, caption, size, dimensions, image, image_mask


def load_index(index, auto_generate):
    if type(index) is not int:
        return None

    path = DATASET.image_paths[index]
    mask_path = DATASET.mask_paths(index)

    img_edit = dict()
    img_edit["composite"] = None
    img_edit["layers"] = None

    img = None
    img_byte_size = 0
    dimensions = ""
    if os.path.exists(path):
        img = Image.open(path)
        img_byte_size = os.path.getsize(path)
        dimensions = f'{img.size[0]} x {img.size[1]}'
        img_edit["background"] = img
    kbyte_size_str = f'{img_byte_size / 1024} kB'

    img_mask = None
    if os.path.exists(mask_path):
        img_mask = Image.open(mask_path)
    elif auto_generate:
        img_mask = ask_mask_from_model(img_edit["background"], 'u2net_human_seg')
        img_mask.save(DATASET.mask_paths(index))

    caption_text = DATASET.read_caption_at(index)

    validation = ""
    if img_mask is None:
        validation += "No mask found.\r\n"
    if img is not None and img_mask is not None:
        edit_width, edit_height = img.size
        mask_width, mask_height = img_mask.size
        if (edit_width, edit_height) != (mask_width, mask_height):
            validation = "Image dimensions do not match!\r\n"
    if img_byte_size < 2E5:
        validation += "Image size is low!\r\n"
    if len(validation) == 0:
        validation = "All good."

    selected_index = index

    return path, kbyte_size_str, dimensions, caption_text, img_edit, img_mask, validation


def navigate_forward(current_index, caption_text, auto_generate_mask):
    if caption_text is not None:
        DATASET.save_caption(current_index, caption_text)

    new_index = min(current_index + 1, DATASET.size() - 1)
    path, caption, size, dimensions, image, image_mask, validation = load_index(new_index, auto_generate_mask)
    return new_index, path, caption, size, dimensions, image, image_mask, validation


def navigate_backward(current_index, caption_text, auto_generate_mask):
    DATASET.save_caption(current_index, caption_text)

    new_index = max(current_index - 1, 0)
    path, caption, size, dimensions, image, image_mask, validation = load_index(new_index, auto_generate_mask)
    return new_index, path, caption, size, dimensions, image, image_mask, validation


def jump(new_index, caption_text, autogenerate_masks):
    if caption_text is not None:
        DATASET.save_caption(selected_index, caption_text)

    path, caption, image_size, image_dims, image, image_mask, validation = load_index(new_index, autogenerate_masks)
    return new_index, path, caption, image_size, image_dims, image, image_mask, validation


def delete_image(current_index, auto_generate_mask):
    if current_index > -1:
        DATASET.delete_image(current_index)

    images_total = DATASET.size()
    if current_index > images_total - 1:
        current_index = max(current_index - 1, 0)
    path, caption, size, dimensions, image, image_mask, validation = load_index(current_index, auto_generate_mask)

    slider_new = gr.Slider(value=current_index, minimum=0, maximum=images_total, label="Image index", step=1,
                           interactive=True)
    return slider_new, path, caption, size, dimensions, image, image_mask, validation, images_total


def generate_caption(current_index, option, subject):
    if not DATASET.initialized:
        return
    path = DATASET.image_paths[current_index]
    if os.path.exists(path):
        if option == 'joytag':
            caption = generate_joytag_caption(path)
        elif option == 'wd14':
            caption = generate_wd14_caption(path)
        elif option == 'florence':
            caption = generate_florence_caption(path)
        elif option == 'multi_sbert':
            caption = generate_multi_sbert(path, CONFIG.sbert_taggers(), CONFIG.sbert_threshold())
        else:
            caption = generate_rag_caption(path)

        if subject is not None:
            caption = f"{subject}, {caption}"
        return caption
    return


def apply_mask_action(mask, image_dict: EditorValue):
    mask = Image.fromarray(mask).convert('RGBA')
    mask = np.array(mask)
    red, green, blue, alpha = mask.T  # Temporarily unpack the bands for readability

    # Replace black with transparent areas (leaves alpha values alone...)
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    mask[...][black_areas.T] = (0, 0, 0, 0)  # Transpose back needed

    image_dict['layers'] = [mask]
    return image_dict


def upscale_image_action(image_dict: EditorValue, progress=gr.Progress()) -> EditorValue:
    img = image_dict['background'].convert("RGB")
    img_out = upscale_image(img, CONFIG.upscaler())

    image_dict['background'] = img_out
    return image_dict


def remove_background_action(image_dict) -> EditorValue:
    img = remove_background(image_dict['background'], 'u2net_human_seg')
    image_dict['background'] = img
    return image_dict


def save_image_action(index, image_dict):
    img = image_dict['background']
    img.save(DATASET.image_paths[index])


def generate_mask(index) -> EditorValue:
    if not DATASET.initialized or not DATASET.mask_support:
        return EditorValue(background=None, layers=[], composite=None)
    path = DATASET.image_paths[index]
    if os.path.exists(path):
        with Image.open(path) as img:
            img.load()

    img_edit = dict()
    img_edit["composite"] = None
    img_edit["layers"] = [ask_mask_from_model(img, 'u2net_human_seg')]
    img_edit["background"] = img
    return img_edit


def save_mask_action(index, editor_value: EditorValue):
    if not DATASET.initialized or not DATASET.mask_support:
        return
    if editor_value["layers"] is not None:
        img_data = editor_value["layers"][0]
        img_data = img_data.convert('RGB')
        print('Saving ', DATASET.mask_paths(index))
        img_data.save(DATASET.mask_paths(index))
        return img_data
    return None


def batch_process(upscale, mask, captioning, subject, progress=gr.Progress(track_tqdm=True)):
    log = ""
    for i in progress.tqdm(range(0, DATASET.size())):
        path, caption, size, dimensions, image, image_mask, validation = load_index(i, False)
        log += "Processing " + path + "\n"
        image = Image.open(path)
        if upscale:
            image = upscale_image(image, CONFIG.upscaler())
            image.save(path)
        if mask:
            mask = ask_mask_from_model(image, 'u2net_human_seg')
            mask.save(DATASET.mask_paths(i))
        if captioning:
            caption = generate_caption(i, 'multi_sbert', subject)
            DATASET.save_caption(i, caption)
    return log


def refresh_thumbnails(tags: list):
    if not DATASET.initialized:
        return
    image_controls = []

    def check_relevance(index, img_path, mask_path, caption):
        image_tags = DATASET.read_tags_at(index)
        # if any element of tags is in image_tags, create a gr.Image object with the respective image
        if any(tag in image_tags for tag in tags):
            # read image as PIL.Image and add it to the list
            image_controls.append(Image.open(img_path))

    DATASET.scan(check_relevance)
    return image_controls


def tab_settings():
    with gr.Tab("Settings"):
        def update_sbert_taggers(taggers):
            CONFIG.update("sbert_taggers", taggers)

        def update_sbert_threshold(value):
            CONFIG.update("sbert_threshold", value)

        def update_upscaler(upscaler):
            CONFIG.update("upscaler", upscaler)

        with gr.Row("Captioning"):
            checkbox_sbert_taggers = gr.CheckboxGroup(choices=['joytag', 'wd14', 'florence'],
                                                      label="SBERT taggers", value=CONFIG.sbert_taggers())
            checkbox_sbert_taggers.change(update_sbert_taggers, inputs=checkbox_sbert_taggers)
            slider_sbert_threshold = gr.Slider(value=CONFIG.sbert_threshold(), minimum=0., maximum=1.0,
                                               label="SBERT threshold", step=0.01)
            slider_sbert_threshold.change(update_sbert_threshold, inputs=slider_sbert_threshold)
        with gr.Row("Upscalers"):
            radio_upscalers = gr.Radio(choices=[upscaler.name for upscaler in Upscalers],
                                          label="Upscaler", value=CONFIG.upscaler())
            radio_upscalers.change(update_upscaler, inputs=radio_upscalers)


def tab_validate():
    with gr.Tab("Validate"):
        with gr.Row():
            checkboxes_aspect_ratios = gr.CheckboxGroup(choices=[aspect.code() for aspect in ImageAspects],
                                                        label="Valid aspect ratios")
            button_validate_dataset = gr.Button(value=process_symbol + " Validate", elem_id="button_execute")
        with gr.Row():
            df_validation = gr.DataFrame()

        button_validate_dataset.click(validate_dataset,
                                      inputs=[checkboxes_aspect_ratios],
                                      outputs=[df_validation])


def tab_captions():
    with gr.Tab("Captions"):
        with gr.Accordion("Tag cloud"):
            with gr.Row():
                with gr.Column(scale=2):
                    button_create_tag_cloud = gr.Button("Refresh cloud")
                    checkbox_tag_cloud = gr.CheckboxGroup(label="Tag cloud", )
                with gr.Column():
                    button_remove_tags = gr.Button("Remove selected tags")
                    with gr.Row():
                        textbox_add_tag = gr.Textbox(placeholder="Enter a new tag", )
                        with gr.Row():
                            button_prepend_tag = gr.Button(value="Prepend tag")
                            button_append_tag = gr.Button(value="Append tag")

            button_create_tag_cloud.click(create_tag_cloud, outputs=checkbox_tag_cloud)
            button_remove_tags.click(remove_selected_tags, inputs=checkbox_tag_cloud, outputs=checkbox_tag_cloud)
            button_prepend_tag.click(prepend_tag, inputs=[textbox_add_tag], outputs=checkbox_tag_cloud)
            button_append_tag.click(append_tag, inputs=[textbox_add_tag], outputs=checkbox_tag_cloud)

            with gr.Row("View"):
                thumb_view = gr.Gallery(allow_preview=True, preview=False, columns=6)

            checkbox_tag_cloud.change(refresh_thumbnails, inputs=checkbox_tag_cloud, outputs=thumb_view)

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


def tab_batch():
    with gr.Tab("Batch"):
        with gr.Row("Processing"):
            with gr.Row():
                checkbox_batch_upscale = gr.Checkbox(label="Upscale", value=False)
                checkbox_batch_mask = gr.Checkbox(label="Mask", value=False)
                checkbox_batch_caption = gr.Checkbox(label="Caption", value=True)
            with gr.Row():
                textbox_processing = gr.Textbox(label="Log", placeholder="Log", lines=1, interactive=False)
                button_batch_process = gr.Button(value=process_symbol + " Batch", elem_id="button_batch")

            inputs = [checkbox_batch_upscale, checkbox_batch_mask, checkbox_batch_caption, textbox_subject]
            button_batch_process.click(batch_process, inputs=inputs, outputs=textbox_processing)


def tab_editing():
    with gr.Tab("Edit"):
        with gr.Row():
            slider = gr.Slider(value=0, minimum=0, maximum=1, label="Image index", step=1, interactive=True)
            textbox_images_total = gr.Textbox(value="0", label="Images total", elem_id="textbox_small")
            button_backward = gr.Button(value=backward_symbol, elem_id='open_folder_small')
            button_forward = gr.Button(value=forward_symbol, elem_id='open_folder_small')
            button_delete = gr.Button(value=delete_symbol, elem_id='open_folder_small')

        with gr.Row():
            with gr.Column():
                image_path_label = gr.Textbox(interactive=False, label="Image path")
                with gr.Row():
                    textbox_image_size = gr.Textbox(interactive=False, label="Size")
                    textbox_image_dimensions = gr.Textbox(interactive=False, label="Dimensions")
            with gr.Column():
                textbox_caption = gr.Textbox(label="Caption", placeholder="Enter caption here..", lines=3,
                                             interactive=True)
                with gr.Column():
                    radio_engine = gr.Radio(choices=['joytag', 'wd14', 'florence', 'rag', 'multi_sbert'],
                                            label="Caption engine", value='multi_sbert')
                button_generate_caption = gr.Button(value="Generate caption", elem_id='generate_caption')

        with gr.Row():
            with gr.Column():
                gr.Markdown("Image")
                image_editor = gr.ImageEditor(interactive=True, type='pil',
                                              brush=gr.Brush(default_size=50, default_color="#ff0000"),
                                              eraser=gr.Eraser(50))
                with gr.Row():
                    gr.Markdown("Modifications")
                    button_remove_background = gr.Button(value="Remove background", elem_id='rem_bg')
                    button_upscale = gr.Button(value="Upscale", elem_id='rem_bg')
                    button_save_image = gr.Button(value="Save image " + save_style_symbol, elem_id='save_image')
                with gr.Row():
                    gr.Markdown("Mask")
                    button_generate_mask = gr.Button(value="Generate mask " + refresh_symbol, elem_id='generate_mask', )
                    button_save_mask = gr.Button(value="Save mask " + save_style_symbol, elem_id='save_mask')
            with gr.Column():
                gr.Markdown("Mask")
                image_mask_preview = gr.Image(interactive=False, label="Saved mask")
                button_apply_mask = gr.Button(value="Apply mask " + document_symbol, elem_id='apply_mask')

        button_open_dataset.click(init_dataset,
                                  inputs=[input_folder_path, mask_folder_path, checkbox_only_missing_captions,
                                          checkbox_autogenerate_mask],
                                  outputs=[slider, textbox_images_total, image_path_label,
                                           textbox_image_size, textbox_image_dimensions, textbox_caption,
                                           image_editor, image_mask_preview])

        control_output_group = [slider, image_path_label, textbox_image_size, textbox_image_dimensions, textbox_caption,
                                image_editor, image_mask_preview]
        button_backward.click(navigate_backward, inputs=[slider, textbox_caption, checkbox_autogenerate_mask],
                              outputs=control_output_group)
        button_forward.click(navigate_forward, inputs=[slider, textbox_caption, checkbox_autogenerate_mask],
                             outputs=control_output_group)
        button_delete.click(delete_image, inputs=[slider, checkbox_autogenerate_mask],
                            outputs=control_output_group + [textbox_images_total])

        slider.input(jump, inputs=[slider, textbox_caption, checkbox_autogenerate_mask], outputs=control_output_group)

        button_generate_caption.click(generate_caption, inputs=[slider, radio_engine, textbox_subject],
                                      outputs=textbox_caption)
        button_upscale.click(upscale_image_action, inputs=image_editor, outputs=image_editor)
        button_remove_background.click(remove_background_action, inputs=image_editor, outputs=image_editor)
        button_save_image.click(save_image_action, inputs=[slider, image_editor])

        button_apply_mask.click(apply_mask_action, inputs=[image_mask_preview, image_editor], outputs=image_editor)
        button_generate_mask.click(generate_mask, inputs=slider, outputs=image_editor, show_progress="full")
        button_save_mask.click(save_mask_action, inputs=[slider, image_editor], outputs=image_mask_preview,
                               show_progress="full")


if __name__ == '__main__':
    css = ""
    if os.path.exists("./style.css"):
        with open(os.path.join("./style.css"), "r", encoding="utf8") as file:
            css = file.read() + "\n"

    with gr.Blocks(css=css) as app:
        with gr.Accordion("Settings"):
            with gr.Row():
                button_open_dir = gr.Button(value=folder_symbol, elem_id='open_folder_small')
                input_folder_path = gr.Textbox(value="", label="Input folder",
                                               placeholder="Directory containing the images")
                button_open_masks_dir = gr.Button(value=folder_symbol, elem_id='open_folder_small')
                mask_folder_path = gr.Textbox(value="", label="Output masks folder",
                                              placeholder="Directory to store masks")
                button_open_dir.click(get_folder_path, inputs=input_folder_path, outputs=input_folder_path)
                button_open_masks_dir.click(get_folder_path, inputs=mask_folder_path, outputs=mask_folder_path)

            with gr.Row():
                with gr.Column():
                    checkbox_only_missing_captions = gr.Checkbox(label="Only images missing captions", value=False)
                    checkbox_autogenerate_mask = gr.Checkbox(label="Auto-generate masks when missing", value=False)
                with gr.Column():
                    textbox_subject = gr.Textbox(label="Subject", placeholder="Subject or leave blank")
            with gr.Row():
                button_open_dataset = gr.Button(value=process_symbol + " Open", elem_id="button_execute")

        tab_editing()
        tab_batch()
        tab_captions()
        tab_validate()
        tab_settings()

    app.launch()
