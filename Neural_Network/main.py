from __future__ import print_function
from math import log10
import torch
import torch.backends.cudnn as cudnn
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader

from DBPN.solver import DBPNTrainer
from DRCN.solver import DRCNTrainer
from EDSR.solver import EDSRTrainer
from FSRCNN.solver import FSRCNNTrainer
from SRCNN.solver import SRCNNTrainer
from SRGAN.solver import SRGANTrainer
from SubPixelCNN.solver import SubPixelTrainer
from VDSR.solver import VDSRTrainer
#from dataset.data import get_training_set, get_test_set
import dataset.div2k

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='srgan', help='choose which model is going to use')

args = parser.parse_args()

class Net(torch.nn.Module):
    def __init__(self, num_channels, base_filter, upscale_factor=2):
        super(Net, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter // 2, out_channels=num_channels * (upscale_factor ** 2), kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
class SRCNNTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(SRCNNTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader

    def build_model(self):
        self.model = Net(num_channels=1, base_filter=64, upscale_factor=self.upscale_factor).to(self.device)
        self.model.weight_init(mean=0.0, std=0.01)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)

    def save_model(self):
        model_out_path = "model_path.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        self.model.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mse = self.criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
            if epoch == self.nEpochs:
                self.save_model()
        
def main():
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    print('===> Loading datasets')
    
    #train_set = get_training_set(args.upscale_factor)
    #test_set = get_test_set(args.upscale_factor)
    #training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    #testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)

    if args.model == 'sub':
        model = SubPixelTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'srcnn':
        model = SRCNNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'vdsr':
        model = VDSRTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'edsr':
        model = EDSRTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'fsrcnn':
        model = FSRCNNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'drcn':
        model = DRCNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'srgan':
        model = SRGANTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'dbpn':
        model = DBPNTrainer(args, training_data_loader, testing_data_loader)
    else:
        raise Exception("the model does not exist")

    model.run()


if __name__ == '__main__':
    main()
