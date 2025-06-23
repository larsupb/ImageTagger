import math
import os
import shutil
import zipfile
from datetime import datetime
from typing import List
from PIL import Image
import imageio

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif')

def is_image(f):
    return f.lower().endswith(IMAGE_EXTENSIONS)

def is_video(f):
    return f.lower().endswith(VIDEO_EXTENSIONS)


def img_to_caption_path(file_path):
    # replace file extension with .txt
    return os.path.splitext(file_path)[0] + ".txt"


def img_to_mask_path(f, mask_path):
    base_name = os.path.basename(f)
    # remove the file extension because masks should always be .png
    mask_file_name = os.path.splitext(base_name)[0] + ".png"
    return os.path.join(mask_path, mask_file_name)


def is_caption_existing(path):
    caption_path = img_to_caption_path(path)
    return os.path.exists(caption_path) and os.path.getsize(caption_path) > 0


def not_filtered(file, ignore_list: list):
    return not any([ignore_pattern in file for ignore_pattern in ignore_list])


def resize_to_target_megapixels(image, megapixels=0.5):
    """
    Resize the image to a target number of megapixels while maintaining the aspect ratio.
    """
    width, height = image.size
    aspect_ratio = width / height

    # Calculate the new dimensions to get close to 1 megapixel
    target_pixels = 1_000_000 * megapixels
    target_height = int(math.sqrt(target_pixels / aspect_ratio))
    target_width = int(target_height * aspect_ratio)

    return image.resize((target_width, target_height))


def load_media(path):
    if is_video(path):
        # read video file and return the 32th frame or the last frame if the video is shorter
        video = imageio.get_reader(path)
        frame_count = video.get_length()
        if frame_count > 32:
            frame = video.get_data(32)
        else:
            frame = video.get_data(frame_count - 1)
        return Image.fromarray(frame)
    elif is_image(path):
        # read image file and return the image
        return Image.open(path)
    return None


class ImageDataSet:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ImageDataSet, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.media_paths = []
        self.images = []
        self.caption_paths = []
        self.caption_texts = dict()
        self.initialized = False
        self.mask_support = False
        self.masks_path = None

    def load(self, path, masks_path=None, only_missing_captions=False, ignore_list: List = None, subdirectories=False,
             load_images=True):
        if ignore_list is None:
            ignore_list = []

        self.media_paths = []
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
                    if is_image(f) or is_video(f):
                        if not_filtered(f, ignore_list) and not is_caption_existing(full_path):
                            image_paths_temp.append(full_path)
                else:
                    if is_image(f) or is_video(f):
                        if not_filtered(f, ignore_list):
                            image_paths_temp.append(full_path)

        self.media_paths = sorted(image_paths_temp)

        self.caption_paths = [img_to_caption_path(f) for f in self.media_paths]

        if load_images:
            # open and rescale images to 0.5 megapixels
            self.images = [resize_to_target_megapixels(load_media(path), 0.5) for path in self.media_paths]

    def prune(self, path, subdirectories=False):
        # scan the directory recursively and remove the captions who do not belong to an image
        for root, dirs, files in os.walk(path):
            # skip subdirectories if parameter subdirectories is False
            if not subdirectories and root != path:
                continue
            for f in files:
                full_path = os.path.join(root, f)
                # continue if file is no txt file
                if not f.lower().endswith('.txt'):
                    continue
                # check if there exists an image file with the same name
                media_path = os.path.splitext(full_path)[0]
                # append potential media file extensions
                media_path = [media_path + ext for ext in VIDEO_EXTENSIONS + IMAGE_EXTENSIONS]
                # remove the caption file if no media file exists
                if not any([os.path.exists(img) for img in media_path]):
                    os.remove(full_path)

    def empty(self):
        return len(self.media_paths) == 0

    def mask_paths(self, index):
        return img_to_mask_path(self.media_paths[index], self.masks_path)

    def size(self):
        return len(self.media_paths)

    def backup(self):
        # create a compressed backup of the dataset
        # take the content of the data directory and compress it
        to_compress = self.media_paths + self.caption_paths
        # create an archive file with the images and captions
        # store it in the data directory
        target_dir = os.path.dirname(self.media_paths[0])
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        with zipfile.ZipFile(os.path.join(target_dir, 'dataset_backup_' + date + '.zip'), 'w') as zipf:
            for f in to_compress:
                if os.path.exists(f):
                    zipf.write(f, os.path.basename(f), compress_type=zipfile.ZIP_DEFLATED)

    def delete_image(self, current_index):
        os.remove(self.media_paths[current_index])
        if os.path.exists(self.mask_paths(current_index)):
            os.remove(self.mask_paths(current_index))
        if os.path.exists(self.caption_paths[current_index]):
            os.remove(self.caption_paths[current_index])

        del self.media_paths[current_index]
        del self.images[current_index]
        del self.caption_paths[current_index]

    def rename_image(self, current_index, offset):
        '''
        Rename to 5 digit number
        '''
        new_name = str(offset + current_index).zfill(5)
        new_image_path = os.path.join(os.path.dirname(self.media_paths[current_index]),
                                      new_name + self.curate_file_extension(current_index))
        new_caption_path = img_to_caption_path(new_image_path)
        new_mask_path = img_to_mask_path(new_image_path, self.masks_path)

        if not os.path.exists(new_image_path):
            os.rename(self.media_paths[current_index], new_image_path)
            if os.path.exists(self.caption_paths[current_index]):
                os.rename(self.caption_paths[current_index], new_caption_path)
            if os.path.exists(self.mask_paths(current_index)):
                os.rename(self.mask_paths(current_index), new_mask_path)

            # update the paths in the dataset
            self.media_paths[current_index] = new_image_path
            self.caption_paths[current_index] = new_caption_path
            return new_name
        else:
            print("File already exists, skipping rename")
            return None

    def curate_file_extension(self, current_index):
        extension = os.path.splitext(self.media_paths[current_index])[1]
        if extension.lower() in ['.jpg', '.jpeg']:
            extension = '.jpg'
        return extension

    def scan(self, validate):
        if not self.initialized:
            raise Exception("Dataset not initialized!")

        output = []
        for i, image_path in enumerate(self.media_paths):
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
        return self.media_paths.index(path)


INSTANCE = ImageDataSet()
