import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from lib.image_dataset import load_media
from lib.upscaling.util import scale_to_megapixels


class FlorenceTagger:
    def __init__(self):
        # Load the Florence2 model and processor
        model_id = 'MiaoshouAI/Florence-2-base-PromptGen-v2.0'
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()

    def predict(self, image: Image, prompt: str):
        """
            Generate a caption for an image using the Florence2 model.
            :param image: The image to generate a caption for
            :param prompt: <GENERATE_PROMPT> <DETAILED_CAPTION> <MORE_DETAILED_CAPTION>
            :return:
            """
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        # Generate the output
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
                output_scores=True,  # Enable this to get the scores
                return_dict_in_generate=True  # Enable this to get the output as a dict
            )

        generated_ids = outputs.sequences
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Convert the generated text into a caption
        caption = self.processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )

        # Return the caption and probabilities for each token
        return list(caption.values())[0]

def generate_florence_caption(image_path: str, prompt: str = '<GENERATE_PROMPT>') -> str:
    image = load_media(image_path)
    # resize to 0.5 megapixels
    image = scale_to_megapixels(image, 0.5)

    captioning = FlorenceTagger()
    caption = captioning.predict(image, prompt)
    return caption
