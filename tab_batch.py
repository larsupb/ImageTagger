import gradio as gr
from PIL import Image

import config
from lib.captioning import generate_caption, save_caption, TAGGERS
from lib.image_dataset import INSTANCE as DATASET
from lib.masking import ask_mask_from_model
from lib.upscaling import upscale_image
from tab_editing import process_symbol
from ui_navigation import load_index


def batch_process(rename, rename_offset, upscale, mask, captioning, tagger, state_dict, progress=gr.Progress(track_tqdm=True)):
    log = ""
    for i in progress.tqdm(range(0, DATASET.size())):
        loader_data = load_index(i)
        log += "Processing " + loader_data['path'] + "\n"
        image = Image.open(loader_data['path'])
        if rename:
            new_path = DATASET.rename_image(i, rename_offset)
            log += "Renamed to " + new_path + "\n"
        if upscale:
            image = upscale_image(image, config.upscaler(state_dict), state_dict)
            image.save(loader_data['path'])
        if mask:
            mask = ask_mask_from_model(image, 'u2net_human_seg')
            mask.save(DATASET.mask_paths(i))
        if captioning:
            caption = generate_caption(i, tagger, state_dict)
            save_caption(i, caption)
    return log


def tab_batch(state: gr.State):
    with gr.Tab("Batch"):
        with gr.Row("Processing"):
            with gr.Column():
                checkbox_rename = gr.Checkbox(label="Rename (5 digits)", value=False)
                rename_offset = gr.Number(label="Offset", minimum=0, maximum=99999)
            with gr.Column():
                checkbox_batch_upscale = gr.Checkbox(label="Upscale", value=False)
            with gr.Column():
                checkbox_batch_mask = gr.Checkbox(label="Mask", value=False)
            with gr.Column():
                checkbox_batch_caption = gr.Checkbox(label="Caption", value=True)
                checkbox_batch_caption_method = gr.Radio(choices=TAGGERS, value='multi_sbert')
            with gr.Row():
                textbox_processing = gr.Textbox(label="Log", placeholder="Log", lines=1, interactive=False)
                button_batch_process = gr.Button(value=process_symbol + " Batch", elem_id="button_batch")

            inputs = [checkbox_rename, rename_offset, checkbox_batch_upscale, checkbox_batch_mask, checkbox_batch_caption,
                      checkbox_batch_caption_method, state]
            button_batch_process.click(batch_process, inputs=inputs, outputs=textbox_processing)
