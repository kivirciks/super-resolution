import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, num_channels, base_filter, upscale_factor=2):
        super(Net, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
            #nn.Sigmoid(),
            #nn.LeakyReLU(),
            #nn.ELU(),
            nn.Tanh(),
            #nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
            #nn.Sigmoid(),
            #nn.LeakyReLU(),
            #nn.ELU(),
            nn.Tanh(),
            #nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter // 2, out_channels=num_channels * (upscale_factor ** 2), kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(upscale_factor)
        )
        # ===========================================================
        # Изменения здесь
        # ===========================================================
        #prune.random_unstructured(self.layers, name="weight", amount=0.1)

    def forward(self, x):
        out = self.layers(x)
        # ===========================================================
        # Изменения здесь
        # ===========================================================
        #prune.random_unstructured(self.layers, name="weight", amount=0.1)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
