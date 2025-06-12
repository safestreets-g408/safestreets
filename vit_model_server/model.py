import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16

class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTClassifier, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.classifier = nn.Linear(self.vit.heads.head.in_features, num_classes)
        self.bbox_regressor = nn.Linear(self.vit.heads.head.in_features, 4)  # x, y, width, height

    def forward(self, x):
        features = self.vit.forward_features(x)
        cls_token = features[:, 0]  # Get the [CLS] token output
        class_output = self.classifier(cls_token)
        bbox_output = self.bbox_regressor(cls_token)  # Predict bounding box coordinates
        return class_output, bbox_output