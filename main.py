import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch.optim as optim
import scipy.io
import os
from PIL import Image
import numpy

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
image_path = './BSR_bsds500/BSR/BSDS500/data/images/train'
truth_path = './BSR_bsds500/BSR/BSDS500/data/groundTruth/train'


BSR_transform = tv.transforms.Compose([
    tv.transforms.Grayscale(),
    tv.transforms.Lambda(lambda x : imrotate(x)),
    tv.transforms.Resize((48,32)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,),(0.5,))
])

train_set = tv.datasets.ImageFolder(root = image_path, transform = BSR_transform)

ground_truth = []
filenames = []

#load ground truths
for filename in sorted(os.listdir(truth_path)):
    mat = scipy.io.loadmat(os.path.join(truth_path, filename))
    mat = mat['groundTruth'][0,0]['Segmentation'][0][0]

    if mat.shape != (481, 321):
        mat = numpy.rot90(mat)

    ground_truth.append(mat)
    filenames.append(filename)

train_gt_set = []

#load ground truths and add to train set
for data, gt in zip(train_set,ground_truth):
    test = torch.randn(1,44,28)
    train_gt_set.append((data[0],test))
    #train_gt_set.append((data[0], gt))

train_loader = torch.utils.data.DataLoader(train_gt_set, batch_size=40, shuffle=True)

net = Net()

criterion = nn.MSELoss()
optimiser = optim.SGD(net.parameters(), lr=0.01)
num_epochs = 5

for epochs in range(num_epochs):
    for data in train_loader:
        inputs, segments = data

        inputs, segments = Variable(inputs), Variable(segments)

        optimiser.zero_grad()

        outputs = net(inputs)
        print(outputs.shape)

        loss = criterion(outputs, segments)

        loss.backward()
        optimiser.step()
