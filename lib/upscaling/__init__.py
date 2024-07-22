import os
import requests
from enum import Enum
from PIL.Image import Image
from config import CONFIG


class Upscalers(Enum):
    NMKD_Siax_200k_4x = ('4x_NMKD-Siax_200k.pth',
                         'https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth?download=true')
    FaceUpDAT_4x = ('4xFaceUpDAT.pth',
                    'https://huggingface.co/tomjackson2023/upscale_models/resolve/3c8d57db3914a46dd851fccf34f806eedca0fea4/4xFaceUpDAT.pth')
    DeJPG_OmniSR_1x = ('1xDeJPG_OmniSR.pth',
                       'https://huggingface.co/tomjackson2023/test/resolve/3f990fb6a1897567dce5e17b37238e4c40e32cc9/1xDeJPG_OmniSR.pth')


def upscale_image(img: Image, upscaler_name: str):
    from .dat_upscaler import upscale_image_dat
    from .spandrel_upscaler import upscale_image_spandrel

    upscaler = next((upscaler for upscaler in Upscalers if upscaler.name == upscaler_name), None)
    if upscaler is None:
        return None
    upscaler_model = os.path.join(CONFIG.models_dir(), 'upscalers', upscaler.value[0])

    # if upscaler_model file does not exist, download it
    if not os.path.exists(upscaler_model):
        print(f"Downloading upscaler model {upscaler_model} from {upscaler.value[1]}")
        r = requests.get(upscaler.value[1], allow_redirects=True)
        open(upscaler_model, 'wb').write(r.content)

    if upscaler == Upscalers.NMKD_Siax_200k_4x:
        return upscale_image_spandrel(img, upscaler_model)
    elif upscaler == Upscalers.DeJPG_OmniSR_1x:
        return upscale_image_dat(img, upscaler_model)
    return upscale_image_dat(img, upscaler_model)
