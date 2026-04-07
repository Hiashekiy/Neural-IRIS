import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


class CorridorEllipseNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.fc = nn.Linear(resnet.fc.in_features, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        out = self.fc(x)

        dx_dy = out[:, 0:2]
        a_b = F.softplus(out[:, 2:4])
        angle_raw = out[:, 4:6]
        angle_norm = F.normalize(angle_raw, p=2, dim=1)

        return torch.cat([dx_dy, a_b, angle_norm], dim=1)


# Backward compatibility for existing imports.
CorridorEllipseNet = CorridorEllipseNet
