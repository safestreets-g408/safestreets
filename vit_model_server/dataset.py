import os
from xml.etree import ElementTree
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class RoadDamageDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.image_dir = os.path.join(base_path, "images")
        self.ann_dir = os.path.join(base_path, "annotations")
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.files = [f for f in os.listdir(self.ann_dir) if f.endswith('.xml')]
        self.data = self._load_data()

    def _parse_annotation(self, path):
        tree = ElementTree.parse(path)
        root = tree.getroot()
        filename = root.find('filename').text
        objects = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            box = obj.find('bndbox')
            bbox = [int(box.find(t).text) for t in ('xmin', 'ymin', 'xmax', 'ymax')]
            objects.append({'class': cls_name, 'bbox': bbox})
        return filename, objects

    def _load_data(self):
        data = []
        for file in self.files:
            ann_path = os.path.join(self.ann_dir, file)
            filename, objects = self._parse_annotation(ann_path)
            if objects:
                cls = max(set([obj['class'] for obj in objects]), key=[obj['class'] for obj in objects].count)
                data.append((filename, cls, objects))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label, _ = self.data[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label