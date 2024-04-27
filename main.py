import os

import PIL.Image
import numpy as np
import gradio as gr

from PIL import Image
from gradio.components.image_editor import EditorValue

from lib.common_gui import get_folder_path
from lib.image_dataset import ImageDataSet

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'  # 📄
process_symbol = u'\u21A9'
backward_symbol = u'\u25C0'
forward_symbol = u'\u25B6'

dataset = ImageDataSet(None, masks_path=None, only_missing_captions=False)


def read_caption(caption_path):
    caption_text = ""
    if not os.path.exists(caption_path):
        with open(caption_path, 'w') as f:
            f.write(caption_text)
    else:
        with open(caption_path, 'r') as f:
            caption_text = f.read()
    dataset.caption_texts[caption_path] = caption_text
    return caption_text


def save_caption(index, new_text):
    global dataset
    if not dataset.initialized or index < 0:
        return

    caption_path = dataset.caption_paths[index]
    if caption_path not in dataset.caption_texts:
        return
    if dataset.caption_texts[caption_path] != new_text:
        with open(caption_path, 'w') as f:
            f.write(new_text)


if __name__ == '__main__':

    css = ""
    if os.path.exists("./style.css"):
        with open(os.path.join("./style.css"), "r", encoding="utf8") as file:
            css = file.read() + "\n"

    with gr.Blocks(css=css) as app:
        with gr.Row():
            button_open_dir = gr.Button(value=folder_symbol, elem_id='open_folder_small')
            input_folder_path = gr.Textbox(value="", label="Input folder",
                                           placeholder="Directory containing the images")
            button_open_masks_dir = gr.Button(value=folder_symbol, elem_id='open_folder_small')
            mask_folder_path = gr.Textbox(value="", label="Output masks folder", placeholder="Directory to store masks")

            checkbox_only_missing_captions = gr.Checkbox(label="Only images missing captions", value=False)
            button_open_dataset = gr.Button(value=process_symbol + " Open")

        with gr.Row():
            slider = gr.Slider(value=0, minimum=0, maximum=len(dataset.images) - 1, label="Image index", step=1,
                               interactive=False)
            button_backward = gr.Button(value=backward_symbol, elem_id='open_folder_small')
            button_forward = gr.Button(value=forward_symbol, elem_id='open_folder_small')

        with gr.Row():
            image_path_label = gr.Textbox(interactive=False, label="Image path")
            caption_display = gr.Textbox(label="Caption", placeholder="Enter caption here..", lines=5, interactive=True)

        with gr.Row():
            with gr.Column():
                gr.Markdown("Image")
                image_editor = gr.ImageEditor(interactive=True, type='pil', brush=gr.Brush(default_size=40))
                button_save_mask = gr.Button(value="Save mask " + save_style_symbol, elem_id='save_mask')
            with gr.Column():
                gr.Markdown("Mask")
                image_mask_preview = gr.Image(interactive=False, label="Saved mask")
                button_load_mask = gr.Button(value="Apply mask " + document_symbol, elem_id='apply_mask')

        button_open_dir.click(get_folder_path, inputs=input_folder_path, outputs=input_folder_path)
        button_open_masks_dir.click(get_folder_path, inputs=mask_folder_path, outputs=mask_folder_path)

        def init_dataset(path, masks_path, filter_missing_captions):
            global dataset
            dataset = ImageDataSet(path, masks_path, filter_missing_captions)
            if not dataset.len() == 0:
                return navigate_forward(-1)
        button_open_dataset.click(init_dataset,
                                  inputs=[input_folder_path, mask_folder_path, checkbox_only_missing_captions],
                                  outputs=[slider, image_path_label, caption_display, image_editor])

        def load_index(index):
            if type(index) is not int:
                return None

            path = dataset.images[index]
            mask_path = dataset.mask_paths(index)

            img_edit = dict()
            img_edit["composite"] = None
            img_edit["layers"] = None
            if os.path.exists(path):
                with Image.open(path) as img:
                    img.load()
                img_edit["background"] = img

            if os.path.exists(mask_path):
                with Image.open(mask_path) as img_mask:
                    img_mask.load()
                #img_edit["layers"] = [img_mask]

            caption_path = dataset.caption_paths[index]
            caption_text = read_caption(caption_path)

            return path, caption_text, img_edit, img_mask

        def navigate_backward(current_index, caption_text):
            save_caption(current_index, caption_text)

            new_index = max(current_index - 1, 0)
            path, caption, image, image_mask = load_index(new_index)
            return new_index, path, caption, image, image_mask
        button_backward.click(navigate_backward, inputs=[slider, caption_display],
                              outputs=[slider, image_path_label, caption_display, image_editor, image_mask_preview])

        def navigate_forward(current_index, caption_text=None):
            if caption_text is not None:
                save_caption(current_index, caption_text)

            new_index = min(current_index + 1, dataset.len()-1)
            path, caption, image, image_mask = load_index(new_index)
            return new_index, path, caption, image, image_mask
        button_forward.click(navigate_forward, inputs=[slider, caption_display],
                             outputs=[slider, image_path_label, caption_display, image_editor, image_mask_preview])


        def save_mask(index, editor_value: EditorValue):
            global dataset
            if not dataset.initialized or not dataset.mask_support:
                return
            if editor_value["layers"] is not None:
                img_data = editor_value["layers"][0]
                print('Saving ', dataset.mask_paths(index))
                img_data.save(dataset.mask_paths(index))

        button_save_mask.click(save_mask, inputs=[slider, image_editor])

    app.launch()
