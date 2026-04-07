import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialPolygonNet(nn.Module):
    def __init__(self, k_dirs: int = 32):
        super().__init__()
        self.k_dirs = k_dirs

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, k_dirs)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.flatten(1)
        out = self.head(feat)
        # Radii should stay positive.
        return F.softplus(out) + 1e-3
