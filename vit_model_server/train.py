import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RoadDamageDataset
from model import ViTClassifier

def train_model(data_path, epochs=5):
    dataset = RoadDamageDataset(data_path)
    class_names = sorted(set(label for _, label, _ in dataset.data))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    dataset.data = [(f, class_to_idx[l], o) for f, l, o in dataset.data]
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ViTClassifier(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), torch.tensor(labels).to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss={running_loss / len(loader)}")
    
    torch.save(model.state_dict(), "model.pth")
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))