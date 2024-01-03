import os
import gradio as gr

from PIL import Image

from lib.common_gui import get_folder_path
from lib.image_dataset import ImageDataSet

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'  # 📄
process_symbol = u'\u21A9'
backward_symbol = u'\u25C0'
forward_symbol = u'\u25B6'


dataset = ImageDataSet(None, False)


def init_dataset(path, filter_missing_captions):
    global dataset
    dataset = ImageDataSet(path, filter_missing_captions)
    return gr.Slider(maximum=dataset.len(), value=0, interactive=True)


def load_index(index):
    path = dataset.images[index]
    image = Image.open(path)
    caption_path = dataset.caption_paths[index]
    caption_text = read_caption(caption_path)

    return path, image, caption_text, index


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


def save_caption(new_text, old_value):
    global dataset

    caption_path = dataset.caption_paths[old_value]
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
            input_folder_path = gr.Textbox(value="", placeholder="Directory containing the images")
            checkbox_only_missing_captions = gr.Checkbox(label="Only images missing captions", value=False)
            button_open_dataset = gr.Button(value=process_symbol + " Open")

        with gr.Row():
            slider = gr.Slider(minimum=-1, maximum=len(dataset.images) - 1, value=-1, label="Image Index", step=1)
            slider_selected_value = gr.Number(visible=False)
            button_backward = gr.Button(value=backward_symbol, elem_id='open_folder_small')
            button_forward = gr.Button(value=forward_symbol, elem_id='open_folder_small')
        with gr.Row():
            with gr.Column():
                gr.Markdown("Image")
                image_path = gr.Textbox(interactive=False, label="Path")
                image_display = gr.Image()
            with gr.Column():
                gr.Markdown("Caption text")
                caption_display = gr.Textbox(placeholder="Caption")

        button_open_dir.click(get_folder_path, inputs=input_folder_path, outputs=input_folder_path)
        button_open_dataset.click(init_dataset, inputs=[input_folder_path, checkbox_only_missing_captions], outputs=slider)

        slider.change(save_caption, inputs=[caption_display, slider_selected_value], outputs=None)
        slider.change(load_index, inputs=slider,
                      outputs=[image_path, image_display, caption_display, slider_selected_value])

        def navigate_backward(current_index) -> int:
            return max(current_index - 1, 0)
        button_backward.click(navigate_backward, inputs=slider, outputs=slider)

        def navigate_forward(current_index) -> int:
            return min(current_index + 1, dataset.len())
        button_forward.click(navigate_forward, inputs=slider, outputs=slider)

    app.launch()
