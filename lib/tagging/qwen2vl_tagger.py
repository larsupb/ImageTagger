import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from lib.image_dataset import load_media
from lib.upscaling.util import scale_to_megapixels

device = "cuda"

class Qwen2VLCaptioner:
    def __init__(self):
        #self.model_id = "createveai/Qwen2-VL-7B-Instruct-abliterated-4bit"
        self.model_id = "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map=device,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "Describe this image. The woman's name is 'SarahF'."},
                ],
            }
        ]

    @torch.no_grad()
    def predict(self, image: Image.Image, max_new_tokens=384, do_sample=True, temperature=0.7, top_k=40):
        # you can resize the image here if it's not fitting to VRAM, or set model max sizes.
        text_prompt = self.processor.apply_chat_template(self.conversation, add_generation_prompt=True)
        image.save("temp.jpg")
        inputs = self.processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.float16):
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                                 temperature=temperature, use_cache=True, top_k=top_k)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return output_text


def generate_qwen2vl_caption(image_path: str):
    image = load_media(image_path)
    # resize to 0.5 megapixels
    image = scale_to_megapixels(image, 0.5)

    captioning = Qwen2VLCaptioner()
    caption = captioning.predict(image)
    return caption
