import torch
from torch import nn
import torch.nn.functional as F 


class AEDenoiser(nn.Module):
    def __init__(self):
        super(AEDenoiser, self).__init__()
        
        #Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) #skip add 2
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) #skip add 1
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.batchnorm7 = nn.BatchNorm2d(1024)


        #Decoder
        self.deconv1 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=(1, 0))
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=(1, 0)) #skip add 1
        self.batchnorm9 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(384, 64, 3, stride=2, padding=1, output_padding=1)
        self.batchnorm10 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.batchnorm11 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(96, 16, 3, stride=2, padding=1, output_padding=(1, 0)) #skip add 2
        self.batchnorm12 = nn.BatchNorm2d(16)
        self.deconv6 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.batchnorm13 = nn.BatchNorm2d(8)
        self.deconv7 = nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):

        #encoder
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x1 = self.batchnorm3(F.relu(self.conv3(x)))
        x = self.batchnorm4(F.relu(self.conv4(x1)))
        x2 = self.batchnorm5(F.relu(self.conv5(x)))
        x = self.batchnorm6(F.relu(self.conv6(x2)))
        x = self.batchnorm7(F.relu(self.conv7(x)))
        

        #decoder
        x = self.batchnorm8(F.relu(self.deconv1(x)))
        x = self.batchnorm9(F.relu(self.deconv2(x)))
        x = torch.cat((x, x2), dim=1)  #add skip
        x = self.batchnorm10(F.relu(self.deconv3(x)))
        x = self.batchnorm11(F.relu(self.deconv4(x)))
        x = torch.cat((x, x1), dim=1) #add skip
        x = self.batchnorm12(F.relu(self.deconv5(x)))
        x = self.batchnorm13(F.relu(self.deconv6(x)))
        x = F.relu(self.deconv7(x))
        return x