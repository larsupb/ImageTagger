from typing import List

import gradio as gr


def tab_browse(tabs: gr.Tabs, control_output_group: List):
    with gr.Tab(id=0, label="Browse"):
        gallery = gr.Gallery(allow_preview=False, preview=False, columns=8)

    return gallery
