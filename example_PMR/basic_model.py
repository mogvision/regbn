import torch
import torch.nn as nn
import torch.nn.functional as F
from ARGS import args


# Create regbn module here *******************************************
from regbn.RegBN import RegBN
from regbn.RegBN_kwargs import regbn_kwargs
RegBN_ = RegBN(**regbn_kwargs).to(torch.device(f"cuda:{args.gpu}"))
# ********************************************************************


# Colored-and-gray-MNIST
class convnet(nn.Module):
    def __init__(self, 
                num_classes=10, 
                in_channel=None, 
                latent_channels=96):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64, latent_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        x = self.maxpool(x)  # 14x14
        x = self.conv2(x)
        x = self.relu(x)  # 14x14
        x = self.conv3(x)
        x = self.relu(x)  # 7x7
        x = self.conv4(x)
        x = self.relu(x)  # 7x7

        feat = x
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)

        return feat


class CGClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.regbn = args.regbn
        n_classes = 10
        latent_channels = 96

        self.gray_net = convnet(in_channel=1,
                    latent_channels=latent_channels, 
                )

        self.colored_net = convnet(in_channel=3, 
                    latent_channels=latent_channels, 
                )
        self.fc_out = nn.Linear(2*latent_channels, n_classes)

    def forward(self, gray, colored, **kwargs):
        # apply regbn
        if self.regbn:
            gray, colored = RegBN_(gray, colored, **kwargs)

        c = self.colored_net(colored)
        c = torch.flatten(c, 1)

        g = self.gray_net(gray)
        g = torch.flatten(g, 1)

        out = self.fc_out(torch.cat((g, c), dim=1))
        return g, c, out

