import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[0, class_idx]
        target.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.uint8(255 * cam)
        cam = np.array(Image.fromarray(cam).resize((224, 224), Image.BICUBIC))

        return cam


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5):
    image = image.resize((224, 224)).convert("RGBA")
    heatmap_colored = plt.cm.jet(heatmap / 255.0)[:, :, :3]
    alpha_channel = np.full((heatmap.shape[0], heatmap.shape[1], 1), np.uint8(255 * alpha))
    heatmap_rgba = np.concatenate([np.uint8(255 * heatmap_colored), alpha_channel], axis=2)

    overlay = Image.alpha_composite(
        image, Image.fromarray(heatmap_rgba, "RGBA")
    )
    return overlay.convert("RGB")


def get_gradcam_layer(model):
    try:
        return model.features[-1]
    except (IndexError, RuntimeError):
        last_conv = None
        for _name, mod in model.named_modules():
            if hasattr(mod, "weight") and mod.weight is not None and mod.weight.dim() == 4:
                last_conv = mod
        if last_conv is not None:
            return last_conv
        return model.features[-1]


def generate_gradcam(model, image_tensor, class_idx=None):
    target_layer = get_gradcam_layer(model)
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(image_tensor, class_idx)

    if class_idx is None:
        with torch.no_grad():
            class_idx = model(image_tensor).argmax(dim=1).item()

    return heatmap, class_idx
