import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


def generate_florence_caption(image_path: str):
    # Load the Florence2 model and processor
    model_id = 'MiaoshouAI/Florence-2-base-PromptGen'
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()

    # Open image
    image = Image.open(image_path)

    prompt = '<GENERATE_PROMPT>'  # '<DETAILED_CAPTION>' <MORE_DETAILED_CAPTION>'
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Generate the output
    with torch.no_grad():
        outputs = model.generate(
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
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Convert the generated text into a caption
    caption = processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height)
    )

    # Return the caption and probabilities for each token
    return list(caption.values())[0]
