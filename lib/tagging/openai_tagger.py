"""
OpenAI-compatible VLM Tagger

Supports any OpenAI-compatible API endpoint including:
- OpenAI (api.openai.com)
- Ollama (localhost:11434)
- OpenRouter, Together.ai, etc.
"""

import base64
import io

from PIL import Image
from openai import OpenAI

from lib.upscaling.util import scale_to_megapixels


class OpenAITagger:
    def __init__(self, api_key: str = "", base_url: str = "http://localhost:11434/v1", model: str = "qwen3-vl:32b"):
        """
        Initialize the OpenAI-compatible tagger.

        Args:
            api_key: API key for the service (can be empty for local Ollama)
            base_url: Base URL for the API endpoint
            model: Model name to use for captioning
        """
        self.model = model
        self.client = OpenAI(
            api_key=api_key if api_key else "ollama",  # Ollama doesn't need a real key
            base_url=base_url
        )

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64 JPEG."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode("utf-8")
        return base64_image

    def predict(self, image_path: str, prompt: str = "Describe this image in detail.", target_megapixels: float = 0.5) -> str:
        """
        Generate a caption for the given image.

        Args:
            image_path: Path to the image file
            prompt: The prompt/instruction for the VLM
            target_megapixels: Target size in megapixels for image resizing (default 0.5)

        Returns:
            Generated caption string
        """
        image = Image.open(image_path).convert("RGB")
        image = scale_to_megapixels(image, target_megapixels)
        base64_image = self._encode_image(image)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024
        )

        return response.choices[0].message.content


def generate_openai_caption(image_path: str, settings: dict) -> str:
    """
    Generate a caption using an OpenAI-compatible VLM API.

    Args:
        image_path: Path to the image file
        settings: Dictionary containing openai_settings with keys:
                  - api_key: API key for the service
                  - base_url: Base URL for the API
                  - model: Model name
                  - prompt: Prompt for caption generation

    Returns:
        Generated caption string
    """
    api_key = settings.get("api_key", "")
    base_url = settings.get("base_url", "http://localhost:11434/v1")
    model = settings.get("model", "qwen3-vl:32b")
    prompt = settings.get("prompt", "Describe this image in detail.")


    tagger = OpenAITagger(api_key=api_key, base_url=base_url, model=model)
    caption = tagger.predict(image_path, prompt=prompt)

    return caption
