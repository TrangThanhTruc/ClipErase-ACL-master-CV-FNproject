import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Any, List
import os

class Model(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name)

    def forward(self, images: List[Any], texts: List[str]) -> Dict[str, torch.Tensor]:
        device = next(self.clip_model.parameters()).device
        
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)

        outputs = self.clip_model(**inputs)
        
        return {
            "logits_per_image": outputs.logits_per_image,
            "logits_per_text": outputs.logits_per_text,
            "image_embeds": outputs.image_embeds,
            "text_embeds": outputs.text_embeds,
        }

class CheckpointHelper:
    def __init__(self):
        self.best_metric = float("-inf")
        self.min_loss = float("inf")

    def save_model_when_max_recall(self, metric: float, model: nn.Module, path: str):
        # Lưu lại checkpoint vào mỗi epoch phục vụ vẽ ảnh Grad-CAM các mốc thời gian
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"[*] Saved checkpoint to: {path}")