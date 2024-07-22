from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AutoModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM


import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image


def generate_blip_caption(image_path):
    # Load the BLIP model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Open image
    image = Image.open(image_path)

    # Preprocess image
    pixel_values = processor(images=image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        # Increase max_length for potentially longer captions
        # Use beam search to explore more diverse captions
        # Adjust temperature for creativity (higher values generate more diverse text)
        output_ids = model.generate(**pixel_values, max_length=50, num_beams=5, temperature=1)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)

    return caption


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

    # # Calculate the probabilities for each generated token
    # scores = outputs.scores  # List of scores (logits) for each generated token
    # probabilities = []
    # num_input_tokens = inputs["input_ids"].size(1)  # Number of input tokens
    #
    # for i, score in enumerate(scores):
    #     # Apply softmax to get the probability distribution
    #     prob_dist = F.softmax(score, dim=-1)
    #     # Get the token ID of the generated token at this position
    #     token_id = generated_ids[0, i]
    #     # Get the probability of the generated token
    #     token_prob = prob_dist[0, token_id].item()
    #     probabilities.append(token_prob)

    # Convert the generated text into a caption
    caption = processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height)
    )

    # Return the caption and probabilities for each token
    return list(caption.values())[0]


def generate_minicpm_caption(image_path: str):
    model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5',
                                      trust_remote_code=True, torch_dtype=torch.float16)
    #model = model.to(device='cuda')
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    # resize the image to 512x512
    image = image.resize((512, 512))

    question = 'What is in the image?'
    msgs = [{'role': 'user', 'content': question}]

    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,  # if sampling=False, beam_search will be used by default
        temperature=0.7,
        # system_prompt='' # pass system_prompt if needed
    )
    print(res)
    ## if you want to use streaming, please make sure sampling=True and stream=True
    ## the model.chat will return a generator
    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
        stream=True
    )

    generated_text = ""
    for new_text in res:
        generated_text += new_text
        print(new_text, flush=True, end='')
        return new_text
