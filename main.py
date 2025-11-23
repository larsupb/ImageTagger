import os

import gradio as gr

import config
from lib.common_gui import get_folder_path
from tab_batch import tab_batch
from tab_browsing import tab_browse
from tab_captions import tab_captions
from tab_editing import tab_editing, align_visibility, set_bookmark
from tab_settings import tab_settings
from tab_tools import tab_tools
from tab_validation import tab_validate
from ui_navigation import init_dataset, load_index, to_control_group, toggle_bookmark
from ui_symbols import folder_symbol, process_symbol

js = '''
function js(){
    window.set_cookie = function(key, value) {
        document.cookie = key+'='+value+'; Path=/; SameSite=Strict';return [value]
    }
}
'''

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

                def input_folder_path_change(value: str):
                    pass
                input_folder_path.change(fn=input_folder_path_change, inputs = [input_folder_path], outputs = [],
                                         js = "(value) => set_cookie('directory_history', value)")

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
            editing_controls = tab_editing(state, gallery)
            control_output_group += [
                editing_controls.slider,
                editing_controls.images_total,
                editing_controls.image_path,
                editing_controls.image_size,
                editing_controls.image_dimensions,
                editing_controls.caption,
                editing_controls.image_editor,
                editing_controls.mask_preview,
                editing_controls.video_display
            ]

            slider = editing_controls.slider
            image_editor = editing_controls.image_editor
            video_display = editing_controls.video_display
            button_bookmark = editing_controls.bookmark_button
            captions_result = tab_captions(state)
            captions_controls = captions_result.controls
            captions_dependency = captions_result.reload_dependency
            gallery_2 = captions_controls.gallery
            tab_validate(state)
            batch_dependency = tab_batch(state)
            tab_tools(state)
            tab_settings(state)

            def on_gallery_click(data: gr.EventData, state_dict: dict):
                new_idx = data._data['index']
                out = (gr.Tabs(selected=1),) + to_control_group(load_index(new_idx, state_dict))
                return out

            gallery.select(on_gallery_click, inputs=[state], outputs=[tabs] + control_output_group). \
               then(align_visibility, inputs=[image_editor, video_display], outputs=[image_editor, video_display]). \
               then(set_bookmark, inputs=[slider, state], outputs=[button_bookmark])

            def on_caption_gallery_click(data: gr.EventData, state_dict: dict):
                new_idx = data._data['index']
                if 'captions_gallery_mapping' in state_dict:
                    new_idx = state_dict['captions_gallery_mapping'][new_idx]
                    out = (gr.Tabs(selected=1),) + to_control_group(load_index(new_idx, state_dict))
                    return out
                return None

            gallery_2.select(on_caption_gallery_click, inputs=[state], outputs=[tabs] + control_output_group). \
               then(align_visibility, inputs=[image_editor, video_display], outputs=[image_editor, video_display]). \
               then(set_bookmark, inputs=[slider, state], outputs=[button_bookmark])

        button_load_ds.click(init_dataset,
                             inputs=[input_folder_path, mask_folder_path,
                                     cb_only_missing_captions, cb_subdirectories, cb_load_gallery, state],
                             outputs=[gallery] + control_output_group + [state])

        # Force reloading the dataset when there were changes made by dependencies (like batch processing)
        for dependency in [captions_dependency, batch_dependency]:
            dependency.then(
                init_dataset,
                inputs=[input_folder_path, mask_folder_path, cb_only_missing_captions,
                        cb_subdirectories, cb_load_gallery, state],
                outputs=[gallery] + control_output_group + [state]
            )

        def get_history(request: gr.Request):
            key = "directory_history"
            if key in request.cookies:
                return request.cookies[key]
            return ""

        app.load(fn=get_history, js=js, outputs=[input_folder_path])  # Load JavaScript function to set cookies

    app.launch()

