import json
import os


def sbert_taggers(state_dict):
    if "sbert_taggers" not in state_dict:
        state_dict["sbert_taggers"] = ['joytag', 'wd14', 'florence']
    return state_dict['sbert_taggers']


def sbert_threshold(state_dict):
    if "sbert_threshold" not in state_dict:
        state_dict["sbert_threshold"] = 0.7
    return state_dict['sbert_threshold']


def upscaler(state_dict):
    if "upscaler" not in state_dict:
        state_dict['upscaler'] = '4x_NMKD-Siax_200k.pth'
    return state_dict['upscaler']


def models_dir(state_dict):
    if "models_dir" not in state_dict:
        state_dict['models_dir'] = 'models'
    return state_dict['models_dir']


def ignore_list(state_dict):
    if "ignore_list" not in state_dict:
        state_dict["ignore_list"] = ['joytag', 'wd14', 'florence']
    return state_dict['ignore_list']


def update(state: dict, key, value):
    state[key] = value
    json.dump(state, open("settings.json", "w"))


def read() -> dict:
    if not os.path.exists("settings.json"):
        json.dump({}, open("settings.json", "w"))
    return json.load(open("settings.json", "r"))


def tagger_instruction(state_dict):
    if "tagger_instruction" not in state_dict:
        state_dict["tagger_instruction"] = "A descriptive caption for this image:\n"
    return state_dict["tagger_instruction"]
