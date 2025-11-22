import tempfile
import os
import shutil

import gradio as gr
import numpy as np
from PIL import Image
from moviepy import VideoFileClip


def extract_frame(video):
    """Extracts the first frame for visual cropping."""
    if video is None:
        return None
    clip = VideoFileClip(video)
    frame = clip.get_frame(0)  # first frame
    image = Image.fromarray(frame)
    out_path = tempfile.mktemp(suffix=".png")
    image.save(out_path)
    clip.close()
    return out_path


def crop_video(video, image_preview):
    """Applies the visible crop region (non-transparent area) to the full video."""
    if video is None or image_preview is None:
        return None

    image_composite = image_preview["composite"]     # cropped visual mask

    # Convert to PIL Image if needed
    if isinstance(image_composite, np.ndarray):
        image_composite = Image.fromarray(image_composite)

    # Convert composite to RGBA (to get alpha or intensity)
    composite_rgba = image_composite.convert("RGBA")
    alpha = np.array(composite_rgba.split()[-1])  # use alpha channel

    # Fallback: if there's no alpha channel variation, use brightness
    if np.all(alpha == 255):
        gray = np.array(composite_rgba.convert("L"))
        mask = gray > 5  # detect non-black area
    else:
        mask = alpha > 0

    # Find bounding box of non-transparent region
    coords = np.argwhere(mask)
    if coords.size == 0:
        # nothing cropped → just return original video
        return video
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)

    # Convert to int for moviepy
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    # Apply to video
    clip = VideoFileClip(video)
    cropped = clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
    out_path = tempfile.mktemp(suffix=".mp4")
    cropped.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)

    clip.close()
    cropped.close()
    return out_path


def prepare_download_file(temp_video_path, original_video_path):
    """Copy temp file to a properly named file for download."""
    if temp_video_path is None:
        return None, "cropped_video.mp4"

    if original_video_path is None:
        download_name = "cropped_video.mp4"
    else:
        original_name = os.path.basename(original_video_path)
        name_without_ext, ext = os.path.splitext(original_name)
        download_name = f"{name_without_ext}_cropped{ext}"

    # Create a new temp file with the proper name
    temp_dir = tempfile.gettempdir()
    download_path = os.path.join(temp_dir, download_name)
    shutil.copy2(temp_video_path, download_path)

    return download_path, download_name

with gr.Blocks() as demo:
    gr.Markdown("### Video Crop Tool (Interactive)")

    # Navigation bar
    with gr.Row():
        nav_step1 = gr.Button("1. Upload Video", variant="primary", scale=1)
        nav_step2 = gr.Button("2. Crop Frame", variant="secondary", scale=1)
        nav_step3 = gr.Button("3. Download", variant="secondary", scale=1)

    with gr.Row():
        step1_col = gr.Column(scale=1)
        step2_col = gr.Column(scale=0, min_width=0)
        step3_col = gr.Column(scale=0, min_width=0)

    with step1_col:
        input_video = gr.Video(label="Upload Video", height=300)
        extract_btn = gr.Button("Load Preview Frame")

    with step2_col:
        preview_image = gr.ImageEditor(label="Crop Area (first frame preview)")
        crop_btn = gr.Button("Apply Crop", visible=False)

    with step3_col:
        output_video = gr.Video(label="Cropped Video", height=300)
        download_btn = gr.DownloadButton("Download Cropped Video", visible=False)

    # Wire up event handlers
    def switch_to_step(step_num):
        """Switch to a specific step and update navigation buttons."""
        if step_num == 1:
            return (
                gr.Column(scale=1),              # expand step 1
                gr.Column(scale=0, min_width=0), # collapse step 2
                gr.Column(scale=0, min_width=0), # collapse step 3
                gr.Button(variant="primary"),    # highlight nav 1
                gr.Button(variant="secondary"),  # unhighlight nav 2
                gr.Button(variant="secondary"),  # unhighlight nav 3
                gr.Button(visible=True),         # show extract_btn
                gr.Button(visible=False),        # hide crop_btn
                gr.DownloadButton(visible=False) # hide download_btn
            )
        elif step_num == 2:
            return (
                gr.Column(scale=0, min_width=0), # collapse step 1
                gr.Column(scale=1),              # expand step 2
                gr.Column(scale=0, min_width=0), # collapse step 3
                gr.Button(variant="secondary"),  # unhighlight nav 1
                gr.Button(variant="primary"),    # highlight nav 2
                gr.Button(variant="secondary"),  # unhighlight nav 3
                gr.Button(visible=False),        # hide extract_btn
                gr.Button(visible=True),         # show crop_btn
                gr.DownloadButton(visible=False) # hide download_btn
            )
        else:  # step 3
            return (
                gr.Column(scale=0, min_width=0), # collapse step 1
                gr.Column(scale=0, min_width=0), # collapse step 2
                gr.Column(scale=1),              # expand step 3
                gr.Button(variant="secondary"),  # unhighlight nav 1
                gr.Button(variant="secondary"),  # unhighlight nav 2
                gr.Button(variant="primary"),    # highlight nav 3
                gr.Button(visible=False),        # hide extract_btn
                gr.Button(visible=False),        # hide crop_btn
                gr.DownloadButton(visible=True)  # show download_btn
            )

    # Navigation button handlers
    nav_step1.click(
        fn=lambda: switch_to_step(1),
        inputs=None,
        outputs=[step1_col, step2_col, step3_col, nav_step1, nav_step2, nav_step3,
                 extract_btn, crop_btn, download_btn]
    )
    nav_step2.click(
        fn=lambda: switch_to_step(2),
        inputs=None,
        outputs=[step1_col, step2_col, step3_col, nav_step1, nav_step2, nav_step3,
                 extract_btn, crop_btn, download_btn]
    )
    nav_step3.click(
        fn=lambda: switch_to_step(3),
        inputs=None,
        outputs=[step1_col, step2_col, step3_col, nav_step1, nav_step2, nav_step3,
                 extract_btn, crop_btn, download_btn]
    )

    def extract_and_collapse(video):
        """Extract frame and expand step 2."""
        frame = extract_frame(video)
        step_updates = switch_to_step(2)
        return (frame,) + step_updates

    extract_btn.click(
        fn=extract_and_collapse,
        inputs=input_video,
        outputs=[preview_image, step1_col, step2_col, step3_col, nav_step1, nav_step2, nav_step3,
                 extract_btn, crop_btn, download_btn]
    )

    def crop_and_prepare_download(video, image_preview):
        """Crop video, prepare download, and expand step 3."""
        cropped_path = crop_video(video, image_preview)
        if cropped_path:
            download_path, download_name = prepare_download_file(cropped_path, video)
            step_updates = switch_to_step(3)
            return (
                cropped_path,
                gr.DownloadButton(
                    label=f"Download {download_name}",
                    value=download_path
                ),
            ) + step_updates
        # If crop failed, stay on step 2
        return (
            None,
            gr.DownloadButton(label="Download Cropped Video"),
            gr.Column(scale=0, min_width=0), # step 1
            gr.Column(scale=1),              # step 2 (stay here)
            gr.Column(scale=0, min_width=0), # step 3
            gr.Button(variant="secondary"),  # nav 1
            gr.Button(variant="primary"),    # nav 2 (stay here)
            gr.Button(variant="secondary"),  # nav 3
            gr.Button(visible=False),        # hide extract_btn
            gr.Button(visible=True),         # show crop_btn
            gr.DownloadButton(visible=False) # hide download_btn
        )

    crop_btn.click(
        fn=crop_and_prepare_download,
        inputs=[input_video, preview_image],
        outputs=[output_video, download_btn, step1_col, step2_col, step3_col, nav_step1, nav_step2, nav_step3,
                 extract_btn, crop_btn, download_btn]
    )

demo.launch()
