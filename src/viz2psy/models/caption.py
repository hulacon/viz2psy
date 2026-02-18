"""Image captioning via Salesforce BLIP."""

import torch
from PIL import Image

from .base import BaseModel


class CaptionModel(BaseModel):
    """Generate natural language captions using BLIP.

    Uses Salesforce's BLIP model for image captioning. Returns a single
    'caption' column with the generated text description of each image.
    """

    name = "caption"

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        max_length: int = 75,
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.max_length = max_length
        self._processor = None

    def load(self) -> None:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )
        self.model = self.model.eval().to(self.device)

    def predict(self, image: Image.Image) -> dict:
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self._processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
            )

        caption = self._processor.decode(output_ids[0], skip_special_tokens=True)
        return {"caption": caption}

    def predict_batch(self, images: list[Image.Image]) -> list[dict]:
        # Ensure RGB for all images
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

        inputs = self._processor(images, return_tensors="pt", padding=True).to(
            self.device
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
            )

        captions = self._processor.batch_decode(output_ids, skip_special_tokens=True)
        return [{"caption": cap} for cap in captions]
