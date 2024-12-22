from jsonformer import Jsonformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib.tagging.florence_tagger import generate_florence_caption
from lib.tagging.wd14_tagger import generate_wd14_caption


def generate_rag_caption(image_path):
    captions = {
        'florence': {'caption': generate_florence_caption(image_path)},
        'wd14': {'caption': generate_wd14_caption(image_path)}
    }
    combined_values = ",".join([value['caption'] for value in captions.values()])
    split = combined_values.split(",")
    split = [word.strip() for word in split]
    combined_values = ','.join(set(split))
    print("combined_values: ", combined_values)

    #model_path = "solidrust/Meta-Llama-3.1-8B-Instruct-abliterated-AWQ"
    model_path = "/home/lars/SD/models/NLP/Mistral-7B-Instruct-v0.3-GGUF"
    gguf_file = "Mistral-7B-Instruct-v0.3.Q6_K.gguf"
    model = AutoModelForCausalLM.from_pretrained(model_path, gguf_file=gguf_file, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, gguf_file=gguf_file, trust_remote_code=True)

    system_message = (
      "Your task is to select relevant tags {relevantTags} from the raw tags, provided by the user. "
      "The other non-relevant tags {nonRelevantTags} are to be excluded."
      "Guidelines:\n"                                                        
      "{relevantTags} are tags describing the person's outfit, accessories and jewelery (what is the subject wearing?)."
      "{relevantTags} are tags describing objects not related to the person."
      "{relevantTags} are tags describing the background or the general environment."
      "For similar tags try to take the most accurate tag and exclude the other ones."
      "{nonRelevantTags} are tags with physical descriptions of the person like breast size, hair color, eye color, hair cut, tattoos, etc. "      
      "{nonRelevantTags} are tags with anatomical descriptions like body parts, e.g. 'arm', 'leg', 'feet' or their orientation like 'arm up', 'legs crossed', etc."
      "{nonRelevantTags} are tags with adjectives like 'seductive', 'sexy', 'serious', 'barefoot', 'mature', 'young', 'beautiful', 'ugly', etc."
    )

    user_message = ("The raw tags for the image are: " + combined_values +
                    "\nReturn the {relevantTags} and {nonRelevantTags} on the following schema.")
    prompt = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message},
    ]
    # concatenate the prompt
    prompt = "\n".join([f"{p['role']}: {p['content']}" for p in prompt])

    json_schema = {
        "type": "object",
        "properties": {
            "relevant_tags": {"type": "string"},
            "non_relevant_tags": {"type": "string"}
        }
    }

    print(prompt)
    jsonformer = Jsonformer(model.model, tokenizer, json_schema, prompt, max_string_token_length=55)
    generated_data = jsonformer()
    print(generated_data)

    # Generate output
    # tokens = (tokenizer(prompt_template.format(system_message=system_message, prompt=prompt), return_tensors='pt').input_ids.cuda())
    #
    # generation_output = model.generate(tokens, max_new_tokens=1024)
    # # Decode output
    # generation_output = tokenizer.batch_decode(generation_output, constraints=constraints, skip_special_tokens=True)
    #
    # # Extract the assistant's response
    # # Find the part that starts after the "<|im_start|>assistant" token
    # assistant_response = generation_output[0].split("<|im_start|>assistant")[-1].strip()
    #
    # # Optionally, remove the ending token if the model includes it
    # assistant_response = assistant_response.split("<|im_end|>")[0].strip()



if __name__ == '__main__':
    generate_rag_caption("/home/lars/SD/datasets/nikki_hd/img_flux/base/00016.png")