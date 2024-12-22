import gradio as gr

from lib.image_aspects import ImageAspects
from lib.validation import validate_dataset
from ui_symbols import process_symbol


def tab_validate():
    with gr.Tab("Validate"):
        with gr.Row():
            button_validate_dataset = gr.Button(value=process_symbol + " Validate", elem_id="button_execute")
        with gr.Row():
            df_validation = gr.DataFrame()

        button_validate_dataset.click(validate_dataset, outputs=[df_validation])
