import gradio as gr
from lib.image_dataset import INSTANCE as DATASET, load_media

# Define CSS for a responsive grid layout
custom_css = """
<style>
    .gallery-container {
        display: grid;
        gap: 10px;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    }
    .gallery-item {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
</style>
"""

def checkable_gallery():
    # Function to load images from paths
    def load_images(paths):
        return [load_media(path) for path in paths]

    # Function to render the gallery with current images and checkboxes
    def render_gallery():
        # Clear existing checkboxes and images
        nonlocal checkboxes
        checkboxes = []
        gallery_rows.children.clear()

        # Recreate the gallery items
        for i, img in enumerate(images):
            with gr.Column(elem_id="gallery-item"):
                checkbox = gr.Checkbox(label=f"Select {i + 1}", value=False)
                checkboxes.append(checkbox)
                image = gr.Image(value=img, type="pil", width=150, height=150, visible=True)
                gallery_rows.add_child(gr.Row([checkbox, image]))

    # Initialize images
    images = load_images([])
    checkboxes = []

    # Create a Gradio Blocks context for the gallery
    with gr.Blocks() as gallery_component:
        # Inject custom CSS
        gr.Markdown(custom_css)

        # Output for checked image IDs
        checked_ids_output = gr.Textbox(label="Checked Image IDs")

        # Placeholder for gallery rows to allow updating
        gallery_rows = gr.Column(elem_id="gallery-container")

        # Render initial gallery
        render_gallery()

        # Function to handle checkbox state change
        def update_checked_images(*checkbox_values):
            checked_ids = [i for i, value in enumerate(checkbox_values) if value]
            return ", ".join(map(str, checked_ids))

        # Attach event listener to each checkbox
        for checkbox in checkboxes:
            checkbox.change(fn=update_checked_images, inputs=checkboxes, outputs=checked_ids_output)

        # Define button to refresh the gallery
        def refresh_gallery():
            # Update the image paths or reload them dynamically if needed
            nonlocal images
            images = load_images(DATASET.media_paths)  # or set image_paths to new paths if dynamic

            # Re-render the gallery with the new images
            render_gallery()
            return checked_ids_output

        # Button to trigger gallery refresh
        refresh_button = gr.Button("Reload Gallery")
        refresh_button.click(refresh_gallery, outputs=[gallery_rows, checked_ids_output])

    # Return the component and the output
    return gallery_component, checked_ids_output, refresh_button
