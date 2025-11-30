import os
import requests
from enum import Enum
from PIL import Image
from typing import Union

import config
from lib.upscaling.util import scale_to_megapixels


class Upscalers(Enum):
    NMKD_Siax_200k_4x = ('4x_NMKD-Siax_200k.pth', 4,
                         'https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth?download=true')
    FaceUpDAT_4x = ('4xFaceUpDAT.pth', 4,
                    'https://huggingface.co/tomjackson2023/upscale_models/resolve/3c8d57db3914a46dd851fccf34f806eedca0fea4/4xFaceUpDAT.pth')
    DeJPG_OmniSR_1x = ('1xDeJPG_OmniSR.pth', 1,
                       'https://huggingface.co/tomjackson2023/test/resolve/3f990fb6a1897567dce5e17b37238e4c40e32cc9/1xDeJPG_OmniSR.pth')
    CGIMasterV1 = ('2x-CGIMaster-v1.pth', 2,
                   'https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-CGIMaster-v1.pth')
    NKMD_PatchySharp = ('4x_NMKD-PatchySharp_240K.pth', 4,),
    RealESRGAN_x4Plus_Anime_6B = ('RealESRGAN_x4plus_anime_6B.pth', 4,
                                  'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth')


def upscale_image(img: Union[Image.Image, str], upscaler_name:str, state_dict: dict, progress=None,
                  max_current_megapixels=1., target_megapixels=2.) -> Image.Image:
    from .spandrel_upscaler import upscale_image_spandrel

    # Load image from path if a string is provided
    if isinstance(img, str):
        img = Image.open(img)

    # skip if image is too large
    if img.width * img.height > max_current_megapixels * 1_000_000:
        print(f"Image's total megapixels count {img.width * img.height / 1_000_000} exceeds maximum megapixels "
              f"count {max_current_megapixels}, skipping upscale.")
        return img

    upscaler = next((upscaler for upscaler in Upscalers if upscaler.name == upscaler_name), None)
    if upscaler is None:
        return None
    # read directory from current file
    upscaler_model = os.path.join(os.getcwd(), config.models_dir(state_dict), 'upscalers', upscaler.value[0])

    # if upscaler_model file does not exist, download it
    if not os.path.exists(upscaler_model):
        print(f"Downloading upscaler model {upscaler_model} from {upscaler.value[2]}")
        r = requests.get(upscaler.value[2], allow_redirects=True)
        open(upscaler_model, 'wb').write(r.content)

    # check that the image has 3 channels (RGB)
    if img.mode != 'RGB':
        print(f"Image mode {img.mode} is not RGB, converting to RGB.")
        img = img.convert('RGB')

    upscaled = upscale_image_spandrel(img, upscaler_model, upscale_factor=upscaler.value[1], progress=progress)

    # if the upscaled image is larger than the target megapixels, resize it
    upscaled = scale_to_megapixels(upscaled, target_megapixels)

    return upscaled
