import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, num_channels, upscale_factor, d=64, s=12, m=4):
        super(Net, self).__init__()
        #self.first_part = nn.Sequential(nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=5, stride=1, padding=2), nn.Sigmoid())
        #self.first_part = nn.Sequential(nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=5, stride=1, padding=2), nn.LeakyReLU())
        self.first_part = nn.Sequential(nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=5, stride=1, padding=2), nn.ELU())
        #self.first_part = nn.Sequential(nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=5, stride=1, padding=2), nn.Tanh())
        #self.first_part = nn.Sequential(nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=5, stride=1, padding=2), nn.PReLU())
        # ===========================================================
        # Изменения здесь
        # ===========================================================
        #prune.random_unstructured(self.first_part, name="weight", amount=0.1)

        self.layers = []
        #self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.Sigmoid())) 
        #self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.LeakyReLU()))
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.ELU()))
        #self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.Tanh()))
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.PReLU()))        
        
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        #self.layers.append(nn.Sigmoid())
        #self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.ELU())
        #self.layers.append(nn.Tanh())
        #self.layers.append(nn.PReLU()) 
        
        #self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.Sigmoid()))
        #self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.LeakyReLU()))
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.ELU()))
        #self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.PReLU()))
        #self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.Tanh()))

        self.mid_part = torch.nn.Sequential(*self.layers)
        # ===========================================================
        # Изменения здесь
        # ===========================================================
        #prune.random_unstructured(self.mid_part, name="weight", amount=0.1)


        # Deconvolution
        self.last_part = nn.ConvTranspose2d(in_channels=d, out_channels=num_channels, kernel_size=9, stride=upscale_factor, padding=3, output_padding=1)
        # ===========================================================
        # Изменения здесь
        # ===========================================================
        #prune.random_unstructured(self.last_part, name="weight", amount=0.1)


    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()
