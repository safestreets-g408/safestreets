import torch
from torchvision import transforms
from PIL import Image
import os
import time
from model import ViTClassifier
from utils import annotate_image

# Load the model and class names once at module initialization
print("Loading ViT model...")
start_time = time.time()

with open("class_names.txt") as f:
    CLASS_NAMES = f.read().splitlines()

MODEL = ViTClassifier(num_classes=len(CLASS_NAMES))
try:
    MODEL.load_state_dict(torch.load("vit_model.pth", map_location="cpu"))
    MODEL.eval()
    print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
except Exception as e:
    print(f"Failed to load model: {e}")
    MODEL = None

# Create transforms once
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_and_annotate(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if MODEL is None:
        raise RuntimeError("Model not loaded successfully at startup")

    print(f"Processing image: {image_path}")
    start_time = time.time()

    try:
        img = Image.open(image_path).convert("RGB")
        original_size = img.size
        input_tensor = TRANSFORM(img).unsqueeze(0)
        
        print("Running inference...")
        with torch.no_grad():
            class_output, bbox_output = MODEL(input_tensor)
            _, predicted = torch.max(class_output, 1)
            pred_class = CLASS_NAMES[predicted.item()]
            
            print(f"Predicted class: {pred_class}")
            
            bbox = bbox_output[0].cpu().numpy()
            width, height = original_size
            x_center, y_center, box_width, box_height = bbox
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height
            
            x1 = max(0, int(x_center - box_width / 2))
            y1 = max(0, int(y_center - box_height / 2))
            x2 = min(width, int(x_center + box_width / 2))
            y2 = min(height, int(y_center + box_height / 2))
            bbox_coords = [x1, y1, x2, y2]

        print("Annotating image...")
        annotated_path = annotate_image(image_path, pred_class, bbox_coords)
        
        print(f"Prediction completed in {time.time() - start_time:.2f} seconds")
        return pred_class, annotated_path
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e
