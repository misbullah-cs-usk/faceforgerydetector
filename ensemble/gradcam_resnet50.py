import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def load_resnet_gradcam(weights_path="ensemble/resnet50_ensemble_mediapipe.pth"):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()

        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze().cpu().detach().numpy()

        cam = np.maximum(cam, 0)
        cam /= cam.max() + 1e-8
        return cam

def generate_gradcam(model, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    target_layer = model.layer4[-1]
    cam_gen = GradCAM(model, target_layer)
    cam = cam_gen.generate(tensor)

    cam_resized = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    heat = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, heat, 0.45, 0)
    return overlay

