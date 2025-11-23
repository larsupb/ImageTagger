import os

import gradio as gr

from lib.image_dataset import ImageDataSet
from tab_editing import process_symbol


def _get_dataset(state_dict: dict) -> ImageDataSet | None:
    """Helper to extract dataset from state dict."""
    if state_dict is None:
        return None
    return state_dict.get('dataset')


def tab_tools(state: gr.State):
    with gr.Tab("Tools"):
        with gr.Column("Copy"):
            with gr.Row():
                copy_options = gr.Radio(choices=['All', 'Bookmarks only'],)
                copy_target_directory = gr.Textbox(label="Target directory", placeholder="Directory to copy images to")
                button_copy = gr.Button(value=process_symbol + " Copy", elem_id="button_execute")
            copy_log = gr.Textbox(label="Log", placeholder="Log output", lines=10, interactive=False)

            def process_copy(copy_options, target_directory, state_dict: dict):
                log = ""
                dataset = _get_dataset(state_dict)
                if dataset is None or not dataset.initialized or dataset.size() == 0:
                    log += "No dataset loaded\n"
                    return log
                # Check if at least parent directory exists for the target directory
                parent_dir = os.path.dirname(target_directory)
                if not os.path.exists(parent_dir):
                    log += f"Parent directory {parent_dir} does not exist\n"
                    return log
                if not os.path.exists(target_directory):
                    try:
                        os.makedirs(target_directory)
                        log += f"Created target directory {target_directory}\n"
                    except Exception as e:
                        log += f"Failed to create target directory {target_directory}: {e}\n"
                        return log

                count = 0
                for i in range(0, dataset.size()):
                    if copy_options == 'Bookmarks only' and not dataset.is_bookmark(i):
                        continue
                    dataset.copy_media(i, target_directory)
                    count += 1
                log += f"Copied {count} images\n"
                return log

            button_copy.click(process_copy, inputs=[copy_options, copy_target_directory, state], outputs=copy_log)
