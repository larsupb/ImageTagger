from typing import List

import torch
#from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from lib.tagging.joytag.joytag import generate_joytag_caption
from lib.tagging.florence_tagger import generate_florence_caption
from lib.tagging.wd14_tagger import generate_wd14_caption


def get_embedding(word, tokenizer, model):
    # Tokenize the word and get the tensor inputs
    inputs = tokenizer(word, return_tensors='pt')

    # Get the hidden states from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # The embeddings are in the last hidden state
    embeddings = outputs.last_hidden_state

    # Pool the embeddings by taking the mean of the token embeddings
    pooled = embeddings.mean(dim=1)
    return pooled.squeeze().numpy()


def find_similar_keywords(embeddings, threshold=0.8):
    words = list(embeddings.keys())
    similar_groups = {}

    for i, word in enumerate(words):
        similar_groups[word] = []
        for j, other_word in enumerate(words):
            if i != j:
                sim = cosine_similarity([embeddings[word]], [embeddings[other_word]])[0][0]
                if sim >= threshold:
                    similar_groups[word].append(other_word)

    return similar_groups


def generate_multi_sbert(image_path, taggers: List, threshold=.8):
    captions = dict()
    for t in taggers:
        if t == 'joytag':
            captions['joytag'] = {'caption': generate_joytag_caption(image_path)}
        elif t == 'wd14':
            captions['wd14'] = {'caption': generate_wd14_caption(image_path)}
        elif t == 'florence':
            captions['florence'] = {'caption': generate_florence_caption(image_path)}

    for key, value in captions.items():
        captions_key_value = f"{key}: {value}\n"
        print(captions_key_value)

    keywords = set()
    for key, value in captions.items():
        # split and strip the caption to get the keywords
        split = value['caption'].split(',')
        split = [word.strip() for word in split]
        keywords.update(split)

    # Load the Sentence-BERT model
    # TODO Reativate
    # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    #
    # # Generate embeddings for each keyword
    # embeddings = model.encode(list(keywords))
    #
    # # Compute cosine similarity matrix
    # cosine_sim_matrix = util.cos_sim(embeddings, embeddings).numpy()
    #
    # # Find and group similar keywords
    # similar_groups = {}
    # for i, word in enumerate(keywords):
    #     similar_groups[word] = []
    #     for j, other_word in enumerate(keywords):
    #         if i != j and cosine_sim_matrix[i][j] >= threshold:
    #             similar_groups[word].append(other_word)
    #
    # # Group similar keywords by merging overlapping groups
    # grouped_keywords = {}
    # visited = set()
    #
    # for word, similar_words in similar_groups.items():
    #     if word not in visited:
    #         group = [word] + similar_words
    #         for w in group:
    #             visited.add(w)
    #         representative = word  # Choose the first word as the representative
    #         grouped_keywords[representative] = group
    # print(grouped_keywords)
    #
    # # Generate the final caption based on the grouped keywords keys
    # final_caption = ', '.join(grouped_keywords.keys())
    #return final_caption
    return None

