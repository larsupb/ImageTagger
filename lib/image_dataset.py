import os
from typing import List

from config import CONFIG


def is_image(f):
    return f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))


def img_to_caption_path(file_path):
    # replace file extension with .txt
    return os.path.splitext(file_path)[0] + ".txt"


def img_to_mask_path(f, mask_path):
    return os.path.join(mask_path, os.path.basename(f))


def is_caption_existing(path):
    caption_path = img_to_caption_path(path)
    return os.path.exists(caption_path) and os.path.getsize(caption_path) > 0


def not_filtered(file):
    return not any([ignore_pattern in file for ignore_pattern in CONFIG.ignore_list()])


class ImageDataSet:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ImageDataSet, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.image_paths = []
        self.caption_paths = []
        self.caption_texts = dict()
        self.initialized = False
        self.mask_support = False
        self.masks_path = None

    def load(self, path, masks_path, only_missing_captions, subdirectories=False):
        self.image_paths = []
        self.caption_paths = []
        self.caption_texts = dict()

        self.initialized = path is not None and os.path.exists(path)

        if masks_path is not None and len(masks_path) > 0:
            if not os.path.exists(masks_path):
                os.makedirs(masks_path)
            self.mask_support = True
        else:
            self.mask_support = False
        self.masks_path = masks_path

        if path is None:
            return

        image_paths_temp = []
        for root, dirs, files in os.walk(path):
            # skip subdirectories if parameter subdirectories is False
            if not subdirectories and root != path:
                continue
            for f in files:
                full_path = os.path.join(root, f)
                if only_missing_captions:
                    if is_image(f) and not_filtered(f) and not is_caption_existing(full_path):
                        image_paths_temp.append(full_path)
                else:
                    if is_image(f) and not_filtered(f):
                        image_paths_temp.append(full_path)

        self.image_paths = sorted(image_paths_temp)

        self.caption_paths = [img_to_caption_path(f) for f in self.image_paths]

    def empty(self):
        return len(self.image_paths) == 0

    def mask_paths(self, index):
        return img_to_mask_path(self.image_paths[index], self.masks_path)

    def size(self):
        return len(self.image_paths)

    def delete_image(self, current_index):
        os.remove(self.image_paths[current_index])
        if os.path.exists(self.mask_paths(current_index)):
            os.remove(self.mask_paths(current_index))
        if os.path.exists(self.caption_paths[current_index]):
            os.remove(self.caption_paths[current_index])

        del self.image_paths[current_index]
        del self.caption_paths[current_index]

    def scan(self, validate):
        if not self.initialized:
            raise Exception("DataSet not initialized!")

        output = []
        for i, image_path in enumerate(self.image_paths):
            output.append(validate(i, image_path, self.mask_paths(i), self.read_caption_at(i)))
        return output

    def read_caption_at(self, index):
        caption_path = self.caption_paths[index]
        caption_text = ""
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                caption_text = f.read()
        self.caption_texts[caption_path] = caption_text
        return caption_text

    def read_tags_at(self, index) -> List[str]:
        caption = self.read_caption_at(index)
        # split the caption into tags
        tags = caption.split(",")
        # trim the tags
        tags = [x.strip() for x in tags]
        # remove empty tags
        tags = [x for x in tags if x]
        return tags

    def save_caption(self, index, new_text):
        if not self.initialized or index < 0:
            return
        caption_path = self.caption_paths[index]
        if caption_path not in self.caption_texts:
            return
        if self.caption_texts[caption_path] != new_text:
            with open(caption_path, 'w') as f:
                f.write(new_text)

    def find_index(self, path):
        return self.image_paths.index(path)


INSTANCE = ImageDataSet()
