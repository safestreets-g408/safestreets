import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RoadDamageDataset
from model import ViTClassifier

class CombinedLoss(nn.Module):
    def __init__(self, cls_weight=1.0, bbox_weight=1.0):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.SmoothL1Loss()  # Also known as Huber loss
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
    
    def forward(self, cls_pred, bbox_pred, cls_target, bbox_target):
        loss_cls = self.cls_loss(cls_pred, cls_target)
        loss_bbox = self.bbox_loss(bbox_pred, bbox_target)
        return self.cls_weight * loss_cls + self.bbox_weight * loss_bbox

def train_model(data_path, epochs=5):
    dataset = RoadDamageDataset(data_path)
    class_names = sorted(set(label for _, label, _ in dataset.data))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    dataset.data = [(f, class_to_idx[l], b) for f, l, b in dataset.data]
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ViTClassifier(num_classes=len(class_names))
    criterion = CombinedLoss(cls_weight=1.0, bbox_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels, bboxes in loader:
            imgs = imgs.to(device)
            labels = torch.tensor(labels).to(device)
            bboxes = bboxes.float().to(device)  # Convert bboxes to float tensor

            optimizer.zero_grad()
            cls_outputs, bbox_outputs = model(imgs)
            loss = criterion(cls_outputs, bbox_outputs, labels, bboxes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss={running_loss / len(loader)}")
    
    torch.save(model.state_dict(), "vit_model.pth")
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))