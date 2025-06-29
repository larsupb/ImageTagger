
from PIL import Image

from lib.tagging.florence_tagger import FlorenceTagger
from lib.upscaling.util import scale_to_megapixels

if __name__ == "__main__":
    # Example usage
    image_path = "00000.png"

    image = Image.open(image_path)
    image = scale_to_megapixels(image)

    #captioning = Qwen2VLCaptioner()
    #caption = captioning.predict(image)
    #print(caption)

    tagger = FlorenceTagger()
    caption = tagger.predict(image, '<DETAILED_CAPTION>')
    print(caption)




