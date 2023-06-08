import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_channels, base_channels, num_residuals):
        super(Net, self).__init__()
        #self.input_conv = nn.Sequential(nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.Sigmoid())
        #self.input_conv = nn.Sequential(nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.LeakyReLU())
        #self.input_conv = nn.Sequential(nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.ELU())
        self.input_conv = nn.Sequential(nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh())
        #self.input_conv = nn.Sequential(nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True))  
        # ===========================================================
        # Изменения здесь
        # ===========================================================
        #prune.random_unstructured(self.input_conv, name="weight", amount=0.1)
        
        #self.residual_layers = nn.Sequential(*[nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.Sigmoid()) for _ in range(num_residuals)])
        #self.residual_layers = nn.Sequential(*[nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.LeakyReLU()) for _ in range(num_residuals)])
        #self.residual_layers = nn.Sequential(*[nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.ELU()) for _ in range(num_residuals)])
        self.residual_layers = nn.Sequential(*[nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh()) for _ in range(num_residuals)])
        #self.residual_layers = nn.Sequential(*[nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True)) for _ in range(num_residuals)])
        # ===========================================================
        # Изменения здесь
        # ===========================================================
        #prune.random_unstructured(self.residual_layers, name="weight", amount=0.1)
        
        self.output_conv = nn.Conv2d(base_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # ===========================================================
        # Изменения здесь
        # ===========================================================
        prune.random_unstructured(self.output_conv, name="weight", amount=0.1)

    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)

    def forward(self, x):
        residual = x
        x = self.input_conv(x)
        x = self.residual_layers(x)
        x = self.output_conv(x)
        x = torch.add(x, residual)
        return x


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
