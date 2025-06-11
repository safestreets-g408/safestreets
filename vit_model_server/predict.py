import torch
from torchvision import transforms
from PIL import Image
import os
from model import ViTClassifier
from utils import annotate_image

def predict_and_annotate(image_path):
    with open("class_names.txt") as f:
        class_names = f.read().splitlines()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    model = ViTClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        pred_class = class_names[predicted.item()]

    annotated_path = annotate_image(image_path, pred_class)
    return pred_class, annotated_path