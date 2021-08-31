import torch
import torch.nn as nn


class Gnet(nn.Module):
    def __init__(self):
        super(Gnet, self).__init__()
        self.ConT_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 512, 4, 2, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 3, 5, 3, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        x = self.ConT_layer(input)
        return x

class Dnet(nn.Module):
    def __init__(self):
        super(Dnet, self).__init__()
        self.Con_layer = nn.Sequential(
            nn.Conv2d(3, 64, 5, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1, 4, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.Con_layer(input).reshape(-1)
        return x

# class DCGAN(nn.Module):
#     def __init__(self):
#         super(DCGAN, self).__init__()
#         self.dnet = Dnet
#         self.gnet = Gnet
#
#     def forward(self, noise):
#         return self.gnet(noise)



# if __name__ == '__main__':
#     G = Gnet()
#     D = Dnet()
#     x1 = torch.randn(2, 3, 96, 96)
#     x2 = torch.randn(2, 64, 1, 1)
#     print(D(x1).shape)
#     print(G(x2).shape)