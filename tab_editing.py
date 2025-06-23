import os
from typing import List

import gradio as gr
import numpy as np
from PIL import Image
from gradio.components.image_editor import EditorValue

import config
from lib.captioning import generate_caption, TAGGERS
from lib.masking import remove_background, ask_mask_from_model
from lib.upscaling import Upscalers, upscale_image
from ui_navigation import load_index, navigate_forward, jump, navigate_backward

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'  # 📄
delete_symbol = u'\u232B'
process_symbol = u'\u21A9'
backward_symbol = u'\u25C0'
forward_symbol = u'\u25B6'

from lib.image_dataset import INSTANCE as DATASET


def delete_image(current_index):
    if current_index > -1:
        DATASET.delete_image(current_index)

    images_total = DATASET.size()
    if current_index > images_total - 1:
        current_index = max(current_index - 1, 0)

    slider_new = gr.Slider(value=current_index, minimum=0, maximum=images_total, label="Image index", step=1,
                           interactive=True)
    # open and rescale images to 0.5 megapixels
    gallery = gr.Gallery(value=DATASET.images, allow_preview=False, preview=False, columns=8, type="pil")

    loader_data = load_index(current_index)
    return gallery, slider_new, *list(loader_data.values())[1:]


def upscale_image_action(image_dict: EditorValue, upscaler_name: str, state_dict: dict,
                         progress=gr.Progress()) -> EditorValue:
    img = image_dict['background'].convert("RGB")
    img_out = upscale_image(img, upscaler_name, state_dict, progress=progress)

    image_dict['background'] = img_out
    return image_dict


def upscale_image_restore_action(index, image_dict: EditorValue) -> EditorValue:
    img = Image.open(DATASET.media_paths[index])
    image_dict['background'] = img
    return image_dict


def remove_background_action(image_dict, state_dict) -> EditorValue:
    img = remove_background(image_dict['background'], 'u2net_human_seg', state_dict)
    image_dict['background'] = img
    return image_dict


def generate_mask(index) -> EditorValue:
    if not DATASET.initialized or not DATASET.mask_support:
        return EditorValue(background=None, layers=[], composite=None)
    path = DATASET.media_paths[index]
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


def apply_mask_action(mask, image_dict: EditorValue):
    mask = Image.fromarray(mask).convert('RGBA')
    mask = np.array(mask)
    red, green, blue, alpha = mask.T  # Temporarily unpack the bands for readability

    # Replace black with transparent areas (leaves alpha values alone...)
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    mask[...][black_areas.T] = (0, 0, 0, 0)  # Transpose back needed

    image_dict['layers'] = [mask]
    return image_dict


def save_image_action(index, image_dict):
    img = image_dict['background']
    img_path = DATASET.media_paths[index]

    if not img_path.endswith('.png'):
        img = img.convert('RGB')
    img.save(img_path)


def tab_editing(state: gr.State, gallery: gr.Gallery):
    with gr.Tab(id=1, label="Edit"):
        with gr.Row():
            slider = gr.Slider(value=0, minimum=0, maximum=1, label="Image index", step=1, interactive=True)
            textbox_images_total = gr.Textbox(value="0", label="Images total", elem_id="textbox_small")
            button_backward = gr.Button(value=backward_symbol, elem_id='open_folder_small')
            button_forward = gr.Button(value=forward_symbol, elem_id='open_folder_small')
            button_delete = gr.Button(value=delete_symbol, elem_id='open_folder_small')

        with gr.Row():
            with gr.Column():
                textbox_image_path = gr.Textbox(interactive=False, label="Image path")
            with gr.Column():
                with gr.Row():
                    textbox_image_size = gr.Textbox(interactive=False, label="Size")
                    textbox_image_dimensions = gr.Textbox(interactive=False, label="Dimensions")

        with gr.Row():
            with gr.Column():
                gr.Markdown("Image")
                image_editor = gr.ImageEditor(interactive=True, type='pil', height=800,
                                              brush=gr.Brush(default_size=50, default_color="#ff0000"),
                                              eraser=gr.Eraser(50))
                with gr.Accordion("Modifications"):
                    with gr.Row():
                        button_remove_background = gr.Button(value="Remove background", elem_id='rem_bg')
                        button_save_image = gr.Button(value="Save image " + save_style_symbol, elem_id='save_image')
                with gr.Accordion("Upscale"):
                    dropdown_upscaler = gr.Dropdown(choices=[upscaler.name for upscaler in Upscalers],
                                                    value=config.upscaler(state.value), show_label=True)
                    with gr.Row():
                        button_upscale = gr.Button(value="Upscale")
                        button_upscale_restore = gr.Button(value="Restore")
                with gr.Accordion("Masking"):
                    with gr.Row():
                        button_generate_mask = gr.Button(value="Generate mask " + refresh_symbol, elem_id='generate_mask', )
                        button_save_mask = gr.Button(value="Save mask " + save_style_symbol, elem_id='save_mask')

            with gr.Column():
                with gr.Tab("Caption"):
                    textbox_caption = gr.Textbox(label="Caption", placeholder="Enter caption here..", lines=3,
                                                 interactive=True)
                    with gr.Column():
                        radio_engine = gr.Radio(choices=TAGGERS,
                                                label="Caption engine", value='multi_sbert')
                    button_generate_caption = gr.Button(value="Generate caption", elem_id='generate_caption')

                with gr.Tab("Mask"):
                    image_mask_preview = gr.Image(interactive=False, label="Saved mask")
                    button_apply_mask = gr.Button(value="Apply mask " + document_symbol, elem_id='apply_mask')

    control_output_group = [slider, textbox_images_total, textbox_image_path, textbox_image_size,
                             textbox_image_dimensions, textbox_caption, image_editor, image_mask_preview]

    button_backward.click(navigate_backward, inputs=[slider, textbox_caption], outputs=control_output_group)
    button_forward.click(navigate_forward, inputs=[slider, textbox_caption], outputs=control_output_group)
    button_delete.click(delete_image, inputs=[slider], outputs=[gallery] + control_output_group)

    slider.input(jump, inputs=[slider, textbox_caption, textbox_image_path], outputs=control_output_group)

    button_generate_caption.click(generate_caption, inputs=[slider, radio_engine, state], outputs=textbox_caption)
    button_upscale.click(upscale_image_action, inputs=[image_editor, state, dropdown_upscaler], outputs=image_editor)
    button_upscale_restore.click(upscale_image_restore_action, inputs=[slider, image_editor], outputs=image_editor)

    button_remove_background.click(remove_background_action, inputs=[image_editor, state], outputs=image_editor)
    button_save_image.click(save_image_action, inputs=[slider, image_editor])

    button_apply_mask.click(apply_mask_action, inputs=[image_mask_preview, image_editor], outputs=image_editor)
    button_generate_mask.click(generate_mask, inputs=slider, outputs=image_editor, show_progress="full")
    button_save_mask.click(save_mask_action, inputs=[slider, image_editor], outputs=image_mask_preview,
                           show_progress="full")

    return control_output_group
