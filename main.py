import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv = nn.Conv2d(1,1,5)

    def forward(self,x):
        x = self.conv(x)
        return x

def imrotate(x):
    if x.size != (481, 321):
        x = x.rotate(90, expand=True)
    return x;

#path to input images
path = './BSR_bsds500/BSR/BSDS500/data/images/train'


BSR_transform = tv.transforms.Compose([
    tv.transforms.Grayscale(),
    tv.transforms.Lambda(lambda x : imrotate(x)),
    tv.transforms.ToTensor()
])

train_set = tv.datasets.ImageFolder(root = path, transform = BSR_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=20, shuffle=True)

for data in train_loader:
    print(data[0].size())
