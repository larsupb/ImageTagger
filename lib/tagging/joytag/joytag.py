import torch
import torch.amp.autocast_mode
import torchvision.transforms.functional as TVF
from PIL import Image

from lib.tagging.joytag.joytag_model import VisionModel

path = '/home/lars/SD/models/tagging/joytag'  # Change this to where you downloaded the model
THRESHOLD = 0.4


with open('top_tags.txt', 'r') as f:
    top_tags = [line.strip() for line in f.readlines() if line.strip()]


def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    # Pad image to square
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # Resize image
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    # Convert to tensor
    image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

    # Normalize
    image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

    return image_tensor


@torch.no_grad()
def predict(image: Image.Image):
    model = VisionModel.load_model(path)
    model.eval()
    model = model.to('cuda')

    image_tensor = prepare_image(image, model.image_size)
    batch = {
        'image': image_tensor.unsqueeze(0).to('cuda'),
    }

    with torch.amp.autocast_mode.autocast('cuda', enabled=True):
        preds = model(batch)
        tag_preds = preds['tags'].sigmoid().cpu()

    scores = {top_tags[i]: tag_preds[0][i] for i in range(len(top_tags))}
    predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
    tag_string = ', '.join(predicted_tags)

    return tag_string, scores


def generate_joytag_caption(image_path: str):
    image = Image.open(image_path)
    tag_string, scores = predict(image)
    #for tag, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    #    print(f'{tag}: {score:.3f}')
    return tag_string
