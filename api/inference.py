import os
import json
import yaml

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.model import get_device
from models.gradcam import generate_gradcam, overlay_heatmap


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


cfg = load_config()
device = get_device()

IMG_SIZE = cfg["data"]["image_size"]
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
eval_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    normalize,
])


class ModelServer:
    def __init__(self):
        self.model = None
        self.gradcam_model = None
        self.class_names = []
        self.loaded = False

    def load_model(self):
        model_path = os.path.join(cfg["paths"]["export_dir"], "model_torchscript.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pth")
            if os.path.exists(model_path):
                self._load_pytorch(model_path)
            else:
                print(f"No model found at {model_path}")
                return
        else:
            self._load_torchscript(model_path)
        self._load_gradcam_model()
        self.loaded = True

    def _load_torchscript(self, path):
        self.model = torch.jit.load(path, map_location=device)
        self.model.eval()
        meta_path = os.path.join(cfg["paths"]["export_dir"], "metadata.json")
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            self.class_names = meta.get("class_names", [])
        print(f"TorchScript model loaded ({len(self.class_names)} classes)")

    def _load_pytorch(self, path):
        from models.model import build_model
        checkpoint = torch.load(path, map_location=device)
        self.class_names = checkpoint.get("class_names", [])
        num_classes = len(self.class_names)
        self.model = build_model(num_classes=num_classes, pretrained=False, dropout=0)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        print(f"PyTorch model loaded ({num_classes} classes)")

    def preprocess(self, image: Image.Image):
        return eval_transform(image).unsqueeze(0).to(device)

    def predict(self, image: Image.Image, top_k: int = 3):
        tensor = self.preprocess(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = probs.topk(top_k)

        top_probs = top_probs.squeeze().cpu().tolist()
        top_indices = top_indices.squeeze().cpu().tolist()

        if isinstance(top_probs, float):
            top_probs = [top_probs]
            top_indices = [top_indices]

        top_3 = [
            {
                "class_name": self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}",
                "probability": round(p, 4),
            }
            for p, idx in zip(top_probs, top_indices)
        ]

        return {
            "class_name": top_3[0]["class_name"],
            "confidence": top_3[0]["probability"],
            "top_3": top_3,
        }

    def _load_gradcam_model(self):
        from models.model import build_model
        ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pth")
        if not os.path.exists(ckpt_path):
            self.gradcam_model = None
            return
        checkpoint = torch.load(ckpt_path, map_location=device)
        num_classes = len(checkpoint.get("class_names", []))
        model = build_model(num_classes=num_classes, pretrained=False, dropout=0)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.train()
        self.gradcam_model = model

    def predict_with_gradcam(self, image: Image.Image):
        tensor = self.preprocess(image)

        if self.gradcam_model is None:
            pred = self.predict(image)
            return {**pred, "heatmap_base64": ""}

        heatmap, class_idx = generate_gradcam(self.gradcam_model, tensor)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence = probs[0, class_idx].item()

        overlay = overlay_heatmap(image, heatmap)
        import io, base64
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "class_name": self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}",
            "confidence": round(confidence, 4),
            "heatmap_base64": b64,
        }

    def num_classes(self):
        return len(self.class_names)


model_server = ModelServer()
