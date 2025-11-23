import os

import gradio as gr

import config
from lib.captioning import generate_caption, save_caption, TAGGERS
from lib.image_dataset import ImageDataSet
from lib.media_cache import generate_thumbnail
from lib.masking import ask_mask_from_model
from lib.upscaling import upscale_image
from tab_editing import process_symbol
from ui_navigation import load_index


def _get_dataset(state_dict: dict) -> ImageDataSet | None:
    """Helper to extract dataset from state dict."""
    if state_dict is None:
        return None
    return state_dict.get('dataset')


def batch_process(rename, rename_offset, upscale, mask, captioning, tagger, state_dict, progress=gr.Progress(track_tqdm=True)):
    log = ""

    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.is_initialized:
        raise Exception("Dataset not initialized!")

    if rename:
        # create a compressed backup of the dataset
        dataset.backup()

    for i in progress.tqdm(range(0, len(dataset))):
        loader_data = load_index(i, state_dict)
        log += "Processing " + loader_data['path'] + "\n"

        item = dataset.get_item(i)
        if item is None:
            continue

        orig_media = generate_thumbnail(item.media_path)
        if rename:
            new_name = dataset.rename_item_numbered(i, rename_offset)
            if new_name is not None:
                log += "Renamed to " + loader_data['path'] + " to  " + new_name + "\n"
                loader_data['path'] = new_name
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
            if dataset.has_mask_support and loader_data['img_mask'] is not None:
                mask_path = item.mask_path
                if mask_path and os.path.exists(mask_path):
                    os.remove(mask_path)
                    loader_data['img_mask'] = None
        if mask:
            generated_mask = ask_mask_from_model(orig_media, 'u2net_human_seg')
            if item.mask_path:
                generated_mask.save(item.mask_path)

        if captioning:
            # TODO: pass media object to captioning
            caption = generate_caption(i, tagger, state_dict)
            save_caption(i, caption, dataset=dataset)

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
                checkbox_batch_caption_method = gr.Radio(choices=TAGGERS, value='joytag')
        with gr.Column():
            textbox_processing = gr.Textbox(label="Log", placeholder="Log", lines=1, interactive=False)
            button_batch_process = gr.Button(value=process_symbol + " Batch", elem_id="button_batch")

        inputs = [checkbox_rename, rename_offset, checkbox_batch_upscale, checkbox_batch_mask,
                  checkbox_batch_caption, checkbox_batch_caption_method, state]
        return button_batch_process.click(batch_process, inputs=inputs, outputs=textbox_processing)
