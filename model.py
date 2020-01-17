import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class Model(nn.Module):
    def __init__(self, feature_dim):
        super(Model, self).__init__()

        self.extractor = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if name == 'maxpool' or name == 'fc':
                continue
            self.extractor.append(module)
        self.extractor = nn.Sequential(*self.extractor)

        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.extractor(x).view(batch_size, -1)
        global_feature = self.fc(feature)
        # [B, D]
        out = F.normalize(global_feature, dim=-1)
        return out
