import PIL.Image
import numpy as np
import torch
import spandrel
from spandrel import ImageModelDescriptor
from PIL.Image import Image

from lib.upscaling.util import image_to_tensor, tensor_to_image


def upscale_image_spandrel(image: Image, model_path):
    # Load the OmniSR model
    loader = spandrel.ModelLoader()
    model = loader.load_from_file(model_path)

    # make sure it's an image to image model
    assert isinstance(model, ImageModelDescriptor)

    # send it to the GPU and put it in inference mode
    model.cuda().eval()

    # convert the image to a tensor
    image_tensor = image_to_tensor(np.array(image))

    # use the model
    with torch.no_grad():
        out = model(image_tensor)

    return PIL.Image.fromarray(tensor_to_image(out))
