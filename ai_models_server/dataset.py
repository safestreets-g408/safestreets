import os
from xml.etree import ElementTree
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class RoadDamageDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.image_dir = os.path.join(base_path, "images")
        self.ann_dir = os.path.join(base_path, "annotations")
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.files = [f for f in os.listdir(self.ann_dir) if f.endswith('.xml')]
        self.data = self._load_data()

    def _parse_annotation(self, path):
        tree = ElementTree.parse(path)
        root = tree.getroot()
        filename = root.find('filename').text
        
        # Get image size for normalization
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        
        objects = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)
            
            # Convert to normalized center coordinates and dimensions
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            
            bbox = [x_center, y_center, box_width, box_height]
            objects.append({'class': cls_name, 'bbox': bbox})
        
        return filename, objects

    def _load_data(self):
        data = []
        for file in self.files:
            ann_path = os.path.join(self.ann_dir, file)
            filename, objects = self._parse_annotation(ann_path)
            if objects:
                # Use the most common class as the label
                cls = max(set([obj['class'] for obj in objects]), key=[obj['class'] for obj in objects].count)
                # Use the bbox of the largest object
                largest_bbox = max(objects, key=lambda x: x['bbox'][2] * x['bbox'][3])['bbox']
                data.append((filename, cls, largest_bbox))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label, bbox = self.data[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, torch.tensor(bbox)