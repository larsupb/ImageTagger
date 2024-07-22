import cv2
import numpy as np
import torch


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] == 1:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img)
    return tensor.unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image * 255.0).round(), 0, 255)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
