import json
import math
import os
import shutil
import zipfile
from datetime import datetime
from io import BytesIO
from typing import List

import PIL
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
    # TODO: Think about absolut path issues
    base_name = os.path.basename(f)
    # remove the file extension because masks should always be .png
    mask_file_name = os.path.splitext(base_name)[0] + ".png"
    return os.path.join(mask_path, mask_file_name)


def is_caption_existing(path):
    caption_path = img_to_caption_path(path)
    return os.path.exists(caption_path) and os.path.getsize(caption_path) > 0


def not_filtered(file, ignore_list: list):
    return not any([ignore_pattern in file for ignore_pattern in ignore_list])


def resize_to_target_megapixels(image: PIL.Image.Image, megapixels=0.5) -> PIL.Image.Image:
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


import imageio
import hashlib
from PIL import Image


def _cache_path(path, max_frames, step, duration) -> str:
    """
    Build a unique cache file path for given video + parameters.
    """
    base_dir = os.getcwd()  # project dir
    cache_dir = os.path.join(base_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    # make a unique hash from params (safe for long paths)
    key = f"{path}-{max_frames}-{step}-{duration}"
    key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key_hash}.gif")


def _resize_and_crop(img, size=128):
    """
    Resize image keeping aspect ratio, then center crop to (size x size).
    """
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    right = left + size
    bottom = top + size

    return img.crop((left, top, right, bottom))

def load_media(path, max_frames=32, step=2, duration=100, size=128) -> str | None:
    """
    Load media and return either an Image (for images) or an animated GIF (for videos).

    Args:
        path (str): Path to the media file.
        max_frames (int): Maximum number of frames to extract from the video.
        step (int): Take every nth frame to reduce GIF size.
        duration (int): Duration of each frame in ms for the GIF.
    """
    cache_file = _cache_path(path, max_frames, step, duration)
    if os.path.exists(cache_file):
        return cache_file

    if is_video(path):
        try:
            # Video was not cached, process it now
            video = imageio.get_reader(path)
            frames = []
            for i, frame in enumerate(video):
                if i % step == 0:
                    img = Image.fromarray(frame)
                    img = _resize_and_crop(img, size=size)
                    frames.append(img)
                if len(frames) >= max_frames:
                    break

            if not frames:
                raise ValueError("No frames extracted")

            # Convert frames to GIF in memory (PIL animated GIF)
            frames[0].info['duration'] = duration
            frames[0].info['loop'] = 0
            gif_bytes = BytesIO()
            frames[0].save(gif_bytes, save_all=True, format="GIF",
                           append_images=frames[1:], duration=100, loop=0)
            gif_bytes.seek(0)

            # also persist to cache folder
            with open(cache_file, "wb") as f:
                f.write(gif_bytes.getbuffer())
            return cache_file
        except Exception as e:
            print(f"Error reading video {path}: {e}")
            return None
    elif is_image(path):
        img = Image.open(path)
        img = resize_to_target_megapixels(img, 0.2)
        cache_file = _cache_path(path, max_frames, step, duration)
        # Save a copy to the cache folder
        img.save(cache_file)
        return cache_file

    return None


class ImageDataSet:
    """
    Manages an image/video dataset with captions, masks, and thumbnails.

    This class is designed to be instantiated per-session and stored in gr.State
    for multi-user Gradio deployments.
    """

    def __init__(self):
        self.base_dir: str = ""
        self.media_paths = []
        self.thumbnail_images = []
        self.caption_paths = []
        self.caption_texts = dict()
        self.initialized = False
        self.mask_support = False
        self.masks_path = None
        self.bookmarks = {}


    def load(self, path, masks_path=None, only_missing_captions=False, ignore_list=None, subdirectories=False,
             load_images=True):
        if ignore_list is None:
            ignore_list = []

        self.media_paths = []
        self.caption_paths = []
        self.caption_texts = {}
        self.bookmarks = {}
        self.initialized = path and os.path.exists(path)

        if not self.initialized:
            return

        self.base_dir = path
        self.mask_support = bool(masks_path)
        self.masks_path = masks_path

        if self.mask_support and not os.path.exists(masks_path):
            os.makedirs(masks_path)

        def should_include(file_path, file_name):
            if not (is_image(file_name) or is_video(file_name)):
                return False
            if not_filtered(file_name, ignore_list):
                if only_missing_captions:
                    return not is_caption_existing(file_path)
                return True
            return False

        for root, dirs, files in os.walk(path):
            if not subdirectories and root != path:
                continue
            for file_name in files:
                full_path = os.path.join(root, file_name)
                if should_include(full_path, file_name):
                    self.media_paths.append(full_path)

        self.media_paths.sort()
        self.caption_paths = [img_to_caption_path(f) for f in self.media_paths]

        if load_images:
            self.thumbnail_images = [load_media(p) for p in self.media_paths]

        bookmarks_path = os.path.join(self.base_dir, "bookmarks.json")
        if os.path.exists(bookmarks_path):
            self.bookmarks = json.load(open(bookmarks_path, "r", encoding="utf8"))

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
        mask_path = img_to_mask_path(self.media_paths[index], self.masks_path)
        return mask_path

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
        del self.thumbnail_images[current_index]
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

    def rename_image_to(self, current_index, new_name):
        """
        Rename an image to a user-specified name (without extension).
        Also renames associated caption and mask files.

        Args:
            current_index: Index of the image in the dataset
            new_name: New filename without extension

        Returns:
            tuple: (success: bool, message: str, new_path: str or None)
        """
        if not self.initialized or current_index < 0 or current_index >= len(self.media_paths):
            return False, "Dataset not initialized or invalid index", None

        # Validate filename
        new_name = new_name.strip()
        if not new_name:
            return False, "Filename cannot be empty", None

        # Check for invalid characters
        invalid_chars = '/\\:*?"<>|'
        if any(c in new_name for c in invalid_chars):
            return False, f"Filename contains invalid characters: {invalid_chars}", None

        current_path = self.media_paths[current_index]
        parent_dir = os.path.dirname(current_path)
        extension = self.curate_file_extension(current_index)

        new_image_path = os.path.join(parent_dir, new_name + extension)
        new_caption_path = img_to_caption_path(new_image_path)

        # Check if target already exists (and it's not the same file)
        if os.path.exists(new_image_path) and new_image_path != current_path:
            return False, "A file with this name already exists", None

        # If renaming to the same name, just return success
        if new_image_path == current_path:
            return True, "No changes made", new_image_path

        try:
            # Rename the image file
            os.rename(current_path, new_image_path)

            # Rename caption file if it exists
            old_caption_path = self.caption_paths[current_index]
            if os.path.exists(old_caption_path):
                os.rename(old_caption_path, new_caption_path)

            # Rename mask file if it exists
            if self.mask_support:
                old_mask_path = self.mask_paths(current_index)
                new_mask_path = img_to_mask_path(new_image_path, self.masks_path)
                if os.path.exists(old_mask_path):
                    os.rename(old_mask_path, new_mask_path)

            # Update caption_texts dict if the old path was cached
            if old_caption_path in self.caption_texts:
                self.caption_texts[new_caption_path] = self.caption_texts.pop(old_caption_path)

            # Update internal paths
            self.media_paths[current_index] = new_image_path
            self.caption_paths[current_index] = new_caption_path

            return True, "File renamed successfully", new_image_path

        except OSError as e:
            return False, f"Error renaming file: {str(e)}", None

    def scan(self, func):
        if not self.initialized:
            raise Exception("Dataset not initialized!")

        output = []
        for i, image_path in enumerate(self.media_paths):
            mask_paths = self.mask_paths(i)
            if not os.path.exists(mask_paths):
                mask_paths = None
            caption_path = self.caption_paths[i]
            if not os.path.exists(caption_path):
                caption_path = None
            output.append(func(i, image_path, mask_paths, caption_path))
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

    def update_image(self, index, new_image):
        if not self.initialized or index < 0 or index >= len(self.media_paths):
            return
        new_image.save(self.media_paths[index])
        self.thumbnail_images[index] = resize_to_target_megapixels(new_image, 0.2)

    def is_bookmark(self, index):
        image_file_name = os.path.basename(self.media_paths[index])
        return image_file_name in self.bookmarks and self.bookmarks[image_file_name]

    def toggle_bookmark(self, index, value: bool):
        image_file_name = os.path.basename(self.media_paths[index])
        self.bookmarks[image_file_name] = value

        with open(os.path.join(self.base_dir, "bookmarks.json"), "w", encoding="utf8") as f:
            json.dump(self.bookmarks, f, indent=4)

    def copy_media(self, index, target_directory):
        if not self.initialized or index < 0 or index >= len(self.media_paths):
            return
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        shutil.copy2(self.media_paths[index], target_directory)
        if os.path.exists(self.caption_paths[index]):
            shutil.copy2(self.caption_paths[index], target_directory)

# Note: The singleton INSTANCE has been removed.
# ImageDataSet instances should now be created per-session and stored in gr.State.
# This enables proper multi-user support in Gradio deployments.
