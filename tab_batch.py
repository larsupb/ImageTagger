import os

import gradio as gr

import config
from lib.captioning import generate_caption, save_caption, TAGGERS
from lib.image_dataset import INSTANCE as DATASET, load_media
from lib.masking import ask_mask_from_model
from lib.upscaling import upscale_image
from tab_editing import process_symbol
from ui_navigation import load_index


def batch_process(rename, rename_offset, upscale, mask, captioning, tagger, state_dict, progress=gr.Progress(track_tqdm=True)):
    log = ""

    if not DATASET.initialized:
        raise Exception("Dataset not initialized!")

    if rename:
        # create a compressed backup of the dataset
        DATASET.backup()

    for i in progress.tqdm(range(0, DATASET.size())):
        loader_data = load_index(i)
        log += "Processing " + loader_data['path'] + "\n"

        orig_media = load_media(loader_data['path'])
        if rename:
            new_path = DATASET.rename_image(i, rename_offset)
            if new_path is not None:
                log += "Renamed to " + loader_data['path'] + " to  " + new_path + "\n"
                loader_data['path'] = new_path
            else:
                log += "Failed to rename " + loader_data['path'] + "\n"
        if upscale:
            image = upscale_image(orig_media, config.upscaler(state_dict), state_dict,
                                  max_current_megapixels=config.upscale_target_megapixels(state_dict),
                                  target_megapixels=config.upscale_target_megapixels(state_dict))
            image.save(loader_data['path'])
            # Refresh loader data
            loader_data['img_edit']['background'] = image
            # TODO: Actually the gradio image component should be updated
            # Remove mask if exists
            if DATASET.mask_support and loader_data['img_mask'] is not None:
                mask_path = DATASET.mask_paths(i)
                if os.path.exists(mask_path):
                    os.remove(mask_path)
                    loader_data['img_mask'] = None
        if mask:
            mask = ask_mask_from_model(orig_media, 'u2net_human_seg')
            mask.save(DATASET.mask_paths(i))

        if captioning:
            # TODO: pass media object to captioning
            caption = generate_caption(i, tagger, state_dict)
            save_caption(i, caption)

    return log


def tab_batch(state: gr.State):
    with (((((gr.Tab("Batch")))))):
        with gr.Column("Processing"):
            with gr.Row():
                checkbox_rename = gr.Checkbox(label="Rename (5 digits)", value=False)
                rename_offset = gr.Number(label="Offset", minimum=0, maximum=99999)
            with gr.Row():
                checkbox_batch_upscale = gr.Checkbox(label="Upscale", value=False)
            with gr.Row():
                checkbox_batch_mask = gr.Checkbox(label="Mask", value=False)
            with gr.Row():
                checkbox_batch_caption = gr.Checkbox(label="Caption", value=True)
                checkbox_batch_caption_method = gr.Radio(choices=TAGGERS, value='multi_sbert')
        with gr.Column():
            textbox_processing = gr.Textbox(label="Log", placeholder="Log", lines=1, interactive=False)
            button_batch_process = gr.Button(value=process_symbol + " Batch", elem_id="button_batch")

        inputs = [checkbox_rename, rename_offset, checkbox_batch_upscale, checkbox_batch_mask,
                  checkbox_batch_caption, checkbox_batch_caption_method, state]
        return button_batch_process.click(batch_process, inputs=inputs, outputs=textbox_processing)
