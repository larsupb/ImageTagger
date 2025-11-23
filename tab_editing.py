import os

import PIL
import gradio as gr
import numpy as np
from PIL import Image
from gradio.components.image_editor import EditorValue
from typing import NamedTuple

import config
from lib.captioning import generate_caption, TAGGERS
from lib.image_dataset import ImageDataSet
from lib.masking import remove_background, ask_mask_from_model
from lib.upscaling import Upscalers, upscale_image
from ui_navigation import load_index, navigate_forward, jump, navigate_backward, to_control_group, toggle_bookmark

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'  # 📄
delete_symbol = u'\u232B'
process_symbol = u'\u21A9'
backward_symbol = u'\u25C0'
forward_symbol = u'\u25B6'
bookmark_on = "🔖"  # symbol when toggled on
bookmark_off = "📑"


def _get_dataset(state_dict: dict) -> ImageDataSet | None:
    """Helper to extract dataset from state dict."""
    if state_dict is None:
        return None
    return state_dict.get('dataset')


class EditingControls(NamedTuple):
    slider: gr.Slider
    images_total: gr.Textbox
    image_path: gr.Textbox
    image_size: gr.Textbox
    image_dimensions: gr.Textbox
    caption: gr.Textbox
    image_editor: gr.ImageEditor
    mask_preview: gr.Image
    video_display: gr.Video
    bookmark_button: gr.Button


def delete_image(current_index, state_dict: dict):
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.initialized:
        return gr.skip()

    if current_index > -1:
        dataset.delete_image(current_index)

    images_total = dataset.size()
    if current_index > images_total - 1:
        current_index = max(current_index - 1, 0)

    slider_new = gr.Slider(value=current_index, minimum=0, maximum=images_total, label="Image index", step=1,
                           interactive=True)
    # open and rescale images to 0.5 megapixels
    gallery = gr.Gallery(value=dataset.thumbnail_images, allow_preview=False, preview=False, columns=8, type="pil")

    loader_data = load_index(current_index, state_dict)
    return gallery, slider_new, *list(loader_data.values())[1:]


def upscale_image_action(img_index: int, image_dict: EditorValue, state_dict: dict, upscaler_name: str,
                         progress=gr.Progress()) -> EditorValue:
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.initialized:
        return image_dict

    img = dataset.media_paths[img_index]
    img_out = upscale_image(img, upscaler_name, state_dict,
                            target_megapixels=config.upscale_target_megapixels(state_dict),
                            max_current_megapixels=config.upscale_target_megapixels(state_dict), progress=progress)
    state_dict["image_upscaled"] = img_out
    state_dict["image_upscaled_index"] = img_index
    image_dict['background'] = img_out
    return image_dict


def upscale_image_restore_action(index, image_dict: EditorValue, state_dict: dict) -> EditorValue:
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.initialized:
        return image_dict

    img = Image.open(dataset.media_paths[index])
    image_dict['background'] = img
    return image_dict


def remove_background_action(image_dict, state_dict) -> EditorValue:
    img = remove_background(image_dict['background'], 'u2net_human_seg', state_dict)
    image_dict['background'] = img
    return image_dict


def generate_mask(index, image_dict: EditorValue, state_dict: dict) -> EditorValue:
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.initialized or not dataset.mask_support:
        return EditorValue(background=None, layers=[], composite=None)

    # Use current editor background to maintain alignment with any modifications (e.g., upscaling)
    img = image_dict.get('background')
    if img is None:
        img = Image.open(dataset.media_paths[index])

    # Load or generate mask
    mask_path = dataset.mask_paths(index)
    if os.path.exists(mask_path):
        mask = Image.open(mask_path)
    else:
        mask = ask_mask_from_model(img, 'u2net_human_seg')

    # Ensure mask dimensions match the background image
    if mask.size != img.size:
        mask = mask.resize(img.size, resample=Image.Resampling.NEAREST)

    return EditorValue(background=img, layers=[mask], composite=None)


def save_mask_action(index, editor_value: EditorValue, state_dict: dict):
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.initialized or not dataset.mask_support:
        return None
    if editor_value["layers"] is not None:
        mask = editor_value["layers"][0]
        mask = mask.convert('RGB')

        print('Saving ', dataset.mask_paths(index))
        # The size of the mask must match the size of the image.
        # So read the image gather its size, resize the mask and save it
        img_orig: PIL.Image = Image.open(dataset.media_paths[index])
        mask.resize(img_orig.size, resample=Image.Resampling.NEAREST)
        mask.save(dataset.mask_paths(index))
        return mask
    return None


def apply_mask_action(mask, image_dict: EditorValue):
    mask = Image.fromarray(mask).convert('RGBA')
    mask = np.array(mask)
    red, green, blue, alpha = mask.T  # Temporarily unpack the bands for readability

    # Replace black with transparent areas (leaves alpha values alone...)
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    mask[...][black_areas.T] = (0, 0, 0, 0)  # Transpose back needed

    image_dict['layers'] = [mask]
    return image_dict


def save_image_action(index, state_dict):
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.initialized:
        return gr.skip()

    if state_dict.get("image_upscaled") and state_dict.get("image_upscaled_index") == index:
        img = state_dict["image_upscaled"]
        dataset.update_image(index, img)

    return to_control_group(load_index(index, state_dict))


def align_visibility(image_editor, video_display):
    has_video = False
    if isinstance(video_display, dict) and video_display.get('value') is not None:
        has_video = True
    elif isinstance(video_display, str):  # Path to video
        has_video = True

    if has_video:
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)


def set_bookmark(index, state_dict: dict):
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.initialized:
        return gr.update(value=bookmark_off)

    if dataset.is_bookmark(index):
        return gr.update(value=bookmark_on)
    else:
        return gr.update(value=bookmark_off)


def start_rename(index, state_dict: dict):
    """Start rename mode: show the filename in an editable textbox."""
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.initialized:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    current_path = dataset.media_paths[index]
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(current_path))[0]

    return (
        gr.update(visible=False),  # Hide path textbox
        gr.update(value=filename, visible=True),  # Show rename textbox with current name
        gr.update(visible=False),  # Hide Rename button
        gr.update(visible=True),   # Show Save button
        gr.update(visible=True),   # Show Cancel button
    )


def execute_rename(index, new_name, state_dict: dict):
    """Execute the rename and return to normal view."""
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.initialized:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    success, message, new_path = dataset.rename_image_to(index, new_name)

    if success:
        display_path = new_path
        gr.Info(message)
    else:
        # Show error in path textbox temporarily
        display_path = f"Error: {message}"
        gr.Warning(message)

    return (
        gr.update(value=display_path, visible=True),  # Show path textbox with result
        gr.update(visible=False),  # Hide rename textbox
        gr.update(visible=True),   # Show Rename button
        gr.update(visible=False),  # Hide Save button
        gr.update(visible=False),  # Hide Cancel button
    )


def cancel_rename(index, state_dict: dict):
    """Cancel rename mode and restore original path display."""
    dataset = _get_dataset(state_dict)
    if dataset is None or not dataset.initialized:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    current_path = dataset.media_paths[index]

    return (
        gr.update(value=current_path, visible=True),  # Show path textbox with original
        gr.update(visible=False),  # Hide rename textbox
        gr.update(visible=True),   # Show Rename button
        gr.update(visible=False),  # Hide Save button
        gr.update(visible=False),  # Hide Cancel button
    )


def tab_editing(state: gr.State, gallery: gr.Gallery):
    with gr.Tab(id=1, label="Edit"):
        with gr.Row():
            slider = gr.Slider(value=0, minimum=0, maximum=1, label="Image index", step=1, interactive=True)
            textbox_images_total = gr.Textbox(value="0", label="Images total", elem_id="textbox_small")
            button_bookmark = gr.Button(value=bookmark_off, elem_id='open_folder_small')
            button_backward = gr.Button(value=backward_symbol, elem_id='open_folder_small')
            button_forward = gr.Button(value=forward_symbol, elem_id='open_folder_small')

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=10):
                        textbox_image_path = gr.Textbox(interactive=False, label="Image path")
                        textbox_rename = gr.Textbox(visible=False, label="New filename")
                    with gr.Column(scale=1, min_width=100):
                        button_rename = gr.Button(value="Rename", elem_id='rename_btn')
                        button_rename_save = gr.Button(value="Save", elem_id='rename_btn', visible=False)
                        button_rename_cancel = gr.Button(value="Cancel", elem_id='rename_btn', visible=False)
                        button_delete = gr.Button(value="Delete", elem_id='rename_btn')
            with gr.Column():
                with gr.Row():
                    textbox_image_size = gr.Textbox(interactive=False, label="Size")
                    textbox_image_dimensions = gr.Textbox(interactive=False, label="Dimensions")

        with gr.Row():
            with gr.Column():
                gr.Markdown("Image/Video")
                image_editor = gr.ImageEditor(interactive=True, type='pil', height=800, fixed_canvas=True,
                                              brush=gr.Brush(default_size=50, default_color="#ff0000"),
                                              eraser=gr.Eraser(50))
                video_display = gr.Video(autoplay=True, loop=True, show_label=False, visible=False, height=800)
                with gr.Accordion("Modifications"):
                    with gr.Row():
                        button_remove_background = gr.Button(value="Remove background", elem_id='rem_bg')
                        button_save_image = gr.Button(value="Save image " + save_style_symbol, elem_id='save_image')
                with gr.Accordion("Upscale"):
                    dropdown_upscaler = gr.Dropdown(choices=[upscaler.name for upscaler in Upscalers],
                                                    value=config.upscaler(state.value), show_label=True)
                    with gr.Row():
                        button_upscale = gr.Button(value="Upscale")
                        button_upscale_restore = gr.Button(value="Restore")
                with gr.Accordion("Masking"):
                    with gr.Row():
                        button_generate_mask = gr.Button(value="Generate mask " + refresh_symbol, elem_id='generate_mask', )
                        button_save_mask = gr.Button(value="Save mask " + save_style_symbol, elem_id='save_mask')

            with gr.Column():
                with gr.Tab("Caption"):
                    textbox_caption = gr.Textbox(label="Caption", placeholder="Enter caption here..", lines=3,
                                                 interactive=True)
                    with gr.Column():
                        radio_engine = gr.Radio(choices=TAGGERS,
                                                label="Caption engine", value='joytag')
                    button_generate_caption = gr.Button(value="Generate caption", elem_id='generate_caption')

                with gr.Tab("Mask"):
                    image_mask_preview = gr.Image(interactive=False, label="Saved mask")
                    button_apply_mask = gr.Button(value="Apply mask " + document_symbol, elem_id='apply_mask')

    control_output_group = [slider, textbox_images_total, textbox_image_path, textbox_image_size,
                             textbox_image_dimensions, textbox_caption, image_editor, image_mask_preview, video_display]

    # Navigation handlers - now include state
    slider.input(jump, inputs=[slider, textbox_caption, textbox_image_path, state], outputs=control_output_group).\
        then(align_visibility, inputs=[image_editor, video_display], outputs=[image_editor, video_display]).\
        then(set_bookmark, inputs=[slider, state], outputs=[button_bookmark])

    button_bookmark.click(toggle_bookmark, inputs=[slider, state], outputs=[button_bookmark])

    button_backward.click(navigate_backward, inputs=[slider, textbox_caption, state], outputs=control_output_group).\
        then(align_visibility, inputs=[image_editor, video_display], outputs=[image_editor, video_display]).\
        then(set_bookmark, inputs=[slider, state], outputs=[button_bookmark])
    button_forward.click(navigate_forward, inputs=[slider, textbox_caption, state], outputs=control_output_group).\
        then(align_visibility, inputs=[image_editor, video_display], outputs=[image_editor, video_display]).\
        then(set_bookmark, inputs=[slider, state], outputs=[button_bookmark])
    button_delete.click(delete_image, inputs=[slider, state], outputs=[gallery] + control_output_group).\
        then(align_visibility, inputs=[image_editor, video_display], outputs=[image_editor, video_display]).\
        then(set_bookmark, inputs=[slider, state], outputs=[button_bookmark])

    # Caption and image operations
    button_generate_caption.click(generate_caption, inputs=[slider, radio_engine, state], outputs=textbox_caption)
    button_upscale.click(upscale_image_action, inputs=[slider, image_editor, state, dropdown_upscaler], outputs=image_editor)
    button_upscale_restore.click(upscale_image_restore_action, inputs=[slider, image_editor, state], outputs=image_editor)

    button_remove_background.click(remove_background_action, inputs=[image_editor, state], outputs=image_editor)
    button_save_image.click(save_image_action, inputs=[slider, state], outputs=control_output_group)

    # Mask operations - now include state
    button_apply_mask.click(apply_mask_action, inputs=[image_mask_preview, image_editor], outputs=image_editor)
    button_generate_mask.click(generate_mask, inputs=[slider, image_editor, state], outputs=image_editor, show_progress="full")
    button_save_mask.click(save_mask_action, inputs=[slider, image_editor, state], outputs=image_mask_preview,
                           show_progress="full")

    # Rename operations
    rename_ui_components = [textbox_image_path, textbox_rename, button_rename, button_rename_save, button_rename_cancel]
    button_rename.click(start_rename, inputs=[slider, state], outputs=rename_ui_components)
    button_rename_save.click(execute_rename, inputs=[slider, textbox_rename, state], outputs=rename_ui_components)
    button_rename_cancel.click(cancel_rename, inputs=[slider, state], outputs=rename_ui_components)

    return EditingControls(
        slider=slider,
        images_total=textbox_images_total,
        image_path=textbox_image_path,
        image_size=textbox_image_size,
        image_dimensions=textbox_image_dimensions,
        caption=textbox_caption,
        image_editor=image_editor,
        mask_preview=image_mask_preview,
        video_display=video_display,
        bookmark_button=button_bookmark
    )
