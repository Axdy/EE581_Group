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


#neural network
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #define the properties of each layer
        self.conv = nn.Conv2d(1,1,5, padding=2)

    def forward(self,x):
        #define the forward pass of the nn
        #back propogation gradients will be automatically calculated
        x = self.conv(x)
        return x

#data set images need to be rotated to portrait
def imrotate(x):
    if x.size != (481, 321):
        x = x.rotate(90, expand=True)
    return x;

#path to input images
image_path = './BSR_bsds500/BSR/BSDS500/data/images/train'
truth_path = './BSR_bsds500/BSR/BSDS500/data/groundTruth/train'

#pretransform dataset images while loading
BSR_transform = tv.transforms.Compose([
    tv.transforms.Grayscale(),
    tv.transforms.Lambda(lambda x : imrotate(x)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,),(0.5,))
])

#load input images from path and apply pertransforms
train_set = tv.datasets.ImageFolder(root = image_path, transform = BSR_transform)

ground_truth = []

#load ground truths
for filename in sorted(os.listdir(truth_path)):
    #extract from .mat files
    mat = scipy.io.loadmat(os.path.join(truth_path, filename))
    mat = mat['groundTruth'][0,0]['Segmentation'][0][0]

    #target data also needs to be portrait
    if mat.shape != (481, 321):
        mat = numpy.rot90(mat)

    #pytorch expects single precision floats
    mat = mat.astype(numpy.float32)
    ground_truth.append(mat)
    
#combined input and target images
train_gt_set = []

#load ground truths and add to train set
for data, gt in zip(train_set,ground_truth):
    train_gt_set.append((data[0], gt))

#create a loader for automatic shuffling and batching
train_loader = torch.utils.data.DataLoader(train_gt_set, batch_size=40, shuffle=True)

#create our nn
net = Net()

#select loss function
criterion = nn.MSELoss()
#create optimiser
optimiser = optim.SGD(net.parameters(), lr=0.01)
#number of times data set will be fully passed through
num_epochs = 5

for epochs in range(num_epochs):
    for data in train_loader:
        inputs, segments = data

        #pytorch needs Variables to perform gradient calculations
        inputs, segments = Variable(inputs), Variable(segments)
        #reset gradients from previous iterations
        optimiser.zero_grad()
        #perform forward pass
        outputs = net(inputs)
        #calculate loss
        loss = criterion(outputs, segments)
        #perform back prop.
        loss.backward()
        #update weights
        optimiser.step()
