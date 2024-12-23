import math

from PIL import Image
from lib.tagging.qwen2vl_tagger import Qwen2VLCaptioner


# Function to resize image to approximately 1 megapixel
def resize_to_target_megapixels(image, megapixels=0.5):
    width, height = image.size
    aspect_ratio = width / height

    # Calculate the new dimensions to get close to desired megapixel
    target_pixels = 1_000_000 * megapixels
    target_height = int(math.sqrt(target_pixels / aspect_ratio))
    target_width = int(target_height * aspect_ratio)

    return image.resize((target_width, target_height), Image.NEAREST)

if __name__ == "__main__":
    # Example usage
    image_path = "/home/lars/SD/datasets/3dmonsterstories/downloads/Droid 447 - An Amazing Journey/amazing_ journey_pg024.jpg"

    image = Image.open(image_path)
    image = resize_to_target_megapixels(image)

    captioning = Qwen2VLCaptioner()
    caption = captioning.predict(image)
    print(caption)




