import torch.nn as nn 
import torch.nn.functional as F 


#Create a residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_residual_block=9):
        super(Generator, self).__init__()

        #stem of generator
        model = [ nn.ReflectionPad2d(3),
        nn.Conv2d(input_channels, 64, 7),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(inplace=True)]

        #downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model +=[nn.Conv2d(in_features, out_feeatures, 3, stride=2, padding=1)
                     nn.InstanceNorm2d(out_features),
                     nn.LeakyReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        #Inserting Residual Blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(in_features)]

        #Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.LeakyReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        #Output layers
        model += [nn.ReflectionPad2d(3),
        nn.Conv2d(64, output_channels, 7),
        nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

        
class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        #Stem
        model = [ nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True)]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True)]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True)]

        #FCN layer
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        #Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)            



