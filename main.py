import os

import gradio as gr

import config
from lib.common_gui import get_folder_path
from tab_batch import tab_batch
from tab_browsing import tab_browse
from tab_captions import tab_captions
from tab_editing import tab_editing
from tab_settings import tab_settings
from tab_validation import tab_validate
from ui_navigation import init_dataset, load_index, to_control_group
from ui_symbols import folder_symbol, process_symbol

if __name__ == '__main__':
    css = ""
    if os.path.exists("./style.css"):
        with open(os.path.join("./style.css"), "r", encoding="utf8") as file:
            css = file.read() + "\n"

    with gr.Blocks(css=css, theme='hmb/amethyst', title='Image Tagger') as app:
        state = gr.State(config.read())

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
                cb_only_missing_captions = gr.Checkbox(label="Only images missing captions", value=False)
                cb_subdirectories = gr.Checkbox(label="Include subdirectories", value=False)
                cb_load_gallery = gr.Checkbox(label="Enable gallery", value=True)
            with gr.Row():
                button_load_ds = gr.Button(value=process_symbol + " Open", elem_id="button_execute")

        control_output_group = []
        with gr.Tabs() as tabs:
            gallery = tab_browse(tabs, control_output_group)
            control_output_group += tab_editing(state, gallery)
            gallery_2 = tab_captions()
            tab_validate()
            tab_batch(state)
            tab_settings(state)

            def on_gallery_click(data: gr.EventData):
                new_idx = data._data['index']
                out = (gr.Tabs(selected=1),) + to_control_group(load_index(new_idx))
                return out
            gallery.select(on_gallery_click, inputs=[], outputs=[tabs] + control_output_group)
            gallery_2.select(on_gallery_click, inputs=[], outputs=[tabs] + control_output_group)

        button_load_ds.click(init_dataset,
                             inputs=[input_folder_path, mask_folder_path,
                                     cb_only_missing_captions, cb_subdirectories, cb_load_gallery, state],
                             outputs=[gallery] + control_output_group)
    app.launch()

