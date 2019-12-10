import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class Net(nn.Module):
    def __init__(self, features_dim=128):
        super(Net, self).__init__()
        self.features_extractor = resnet18()
        self.features_extractor.fc = nn.Linear(512, features_dim)

    def forward(self, x):
        features = self.features_extractor(x)
        features = F.normalize(features)
        return features
