import json
import os


class ConfigClass:
    _instance = None

    settings = dict()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigClass, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # open settings.json or create empty if not exists
        if not os.path.exists("settings.json"):
            json.dump({}, open("settings.json", "w"))
        self.settings = json.load(open("settings.json", "r"))

        if "sbert_taggers" not in self.settings:
            self.settings["sbert_taggers"] = ['joytag', 'wd14', 'florence']
        if "sbert_threshold" not in self.settings:
            self.settings["sbert_threshold"] = 0.65
        if "models_dir" not in self.settings:
            self.settings['models_dir'] = 'models'
        if "ignore_list" not in self.settings:
            self.settings['ignore_list'] = ['masklabel']
        if "upscaler" not in self.settings:
            self.settings['upscaler'] = '4x_NMKD-Siax_200k.pth'

        # save settings
        json.dump(self.settings, open("settings.json", "w"))

    def sbert_taggers(self):
        return self.settings['sbert_taggers']

    def sbert_threshold(self):
        return self.settings['sbert_threshold']

    def upscaler(self):
        return self.settings['upscaler']

    def models_dir(self):
        return self.settings['models_dir']

    def ignore_list(self):
        return self.settings['ignore_list']

    def update(self, key, value):
        self.settings[key] = value
        json.dump(self.settings, open("settings.json", "w"))


CONFIG = ConfigClass()

