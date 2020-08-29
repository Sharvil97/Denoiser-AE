import torch 
import torch.nn as nn
import torchvision.models as models

class Generator(nn.Module):
    def __init__(self, channels=1, kernel_size=3, padding=1, features=128, num_layers = 16):
        super(Generator, self).__init__()
        kernel_size = kernel_size
        padding = padding
        features = features
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):
        x = models.inception_v3(pretrained=True, progress=False)
        x = self.sigmoid(x)
        return x
