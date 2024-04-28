import os


def is_image(f):
    return f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))


def img_to_caption_path(f):
    return f.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt').replace('.gif', '.txt')

def img_to_mask_path(f, mask_path):
    return os.path.join(mask_path, os.path.basename(f))

def is_caption_existing(path):
    return os.path.exists(img_to_caption_path(path))


class ImageDataSet:
    def __init__(self, path, masks_path, only_missing_captions):
        self.images = []
        self.caption_paths = []
        self.caption_texts = dict()

        self.initialized = path is not None and os.path.exists(path)

        self.mask_support = masks_path is not None and os.path.exists(masks_path)
        self.masks_path = masks_path

        if path is None:
            return

        files = os.listdir(path)
        if only_missing_captions:
            self.images = sorted([os.path.join(path, f) for f in files
                                  if is_image(f) and not is_caption_existing(os.path.join(path, f))])
        else:
            self.images = sorted([os.path.join(path, f) for f in files if is_image(f)])

        self.caption_paths = [img_to_caption_path(f) for f in self.images]

    def empty(self):
        return len(self.images) == 0

    def mask_paths(self, index):
        return img_to_mask_path(self.images[index], self.masks_path)

    def size(self):
        return len(self.images) - 1

