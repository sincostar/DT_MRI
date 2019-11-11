# u-net.py:
from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch
from numpy.linalg import svd
from numpy.random import normal
from math import sqrt


class UNet(nn.Module):
    def __init__(self,colordim =1):
        super(UNet, self).__init__()
        self.conv1_1 = nn.Conv2d(colordim, 64, 3)  # input of (n,n,1), output of (n-2,n-2,64)
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(512, 1024, 3)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3)
        self.upconv5 = nn.Conv2d(1024, 512, 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn5_out = nn.BatchNorm2d(1024)
        self.conv6_1 = nn.Conv2d(1024, 512, 3)
        self.conv6_2 = nn.Conv2d(512, 512, 3)
        self.upconv6 = nn.Conv2d(512, 256, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn6_out = nn.BatchNorm2d(512)
        self.conv7_1 = nn.Conv2d(512, 256, 3)
        self.conv7_2 = nn.Conv2d(256, 256, 3)
        self.upconv7 = nn.Conv2d(256, 128, 1)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn7_out = nn.BatchNorm2d(256)
        self.conv8_1 = nn.Conv2d(256, 128, 3)
        self.conv8_2 = nn.Conv2d(128, 128, 3)
        self.upconv8 = nn.Conv2d(128, 64, 1)
        self.bn8 = nn.BatchNorm2d(64)
        self.bn8_out = nn.BatchNorm2d(128)
        self.conv9_1 = nn.Conv2d(128, 64, 3)
        self.conv9_2 = nn.Conv2d(64, 64, 3)
        self.conv9_3 = nn.Conv2d(64, colordim, 1)
        self.bn9 = nn.BatchNorm2d(colordim)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self._initialize_weights()

    def forward(self, x1):
        x1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(x1)))))
        # print('x1 size: %d'%(x1.size(2)))
        x2 = F.relu(self.bn2(self.conv2_2(F.relu(self.conv2_1(self.maxpool(x1))))))
        # print('x2 size: %d'%(x2.size(2)))
        x3 = F.relu(self.bn3(self.conv3_2(F.relu(self.conv3_1(self.maxpool(x2))))))
        # print('x3 size: %d'%(x3.size(2)))
        x4 = F.relu(self.bn4(self.conv4_2(F.relu(self.conv4_1(self.maxpool(x3))))))
        # print('x4 size: %d'%(x4.size(2)))
        xup = F.relu(self.conv5_2(F.relu(self.conv5_1(self.maxpool(x4)))))  # x5
        # print('x5 size: %d'%(xup.size(2)))

        xup = self.bn5(self.upconv5(self.upsample(xup)))  # x6in
        cropidx = (x4.size(2) - xup.size(2)) // 2
        x4 = x4[:, :, cropidx:cropidx + xup.size(2), cropidx:cropidx + xup.size(2)]
        # print('crop1 size: %d, x9 size: %d'%(x4crop.size(2),xup.size(2)))
        xup = self.bn5_out(torch.cat((x4, xup), 1))  # x6 cat x4
        xup = F.relu(self.conv6_2(F.relu(self.conv6_1(xup))))  # x6out

        xup = self.bn6(self.upconv6(self.upsample(xup)))  # x7in
        cropidx = (x3.size(2) - xup.size(2)) // 2
        x3 = x3[:, :, cropidx:cropidx + xup.size(2), cropidx:cropidx + xup.size(2)]
        # print('crop1 size: %d, x9 size: %d'%(x3crop.size(2),xup.size(2)))
        xup = self.bn6_out(torch.cat((x3, xup), 1) ) # x7 cat x3
        xup = F.relu(self.conv7_2(F.relu(self.conv7_1(xup))))  # x7out

        xup = self.bn7(self.upconv7(self.upsample(xup)) ) # x8in
        cropidx = (x2.size(2) - xup.size(2)) // 2
        x2 = x2[:, :, cropidx:cropidx + xup.size(2), cropidx:cropidx + xup.size(2)]
        # print('crop1 size: %d, x9 size: %d'%(x2crop.size(2),xup.size(2)))
        xup = self.bn7_out(torch.cat((x2, xup), 1))  # x8 cat x2
        xup = F.relu(self.conv8_2(F.relu(self.conv8_1(xup))))  # x8out

        xup = self.bn8(self.upconv8(self.upsample(xup)) ) # x9in
        cropidx = (x1.size(2) - xup.size(2)) // 2
        x1 = x1[:, :, cropidx:cropidx + xup.size(2), cropidx:cropidx + xup.size(2)]
        # print('crop1 size: %d, x9 size: %d'%(x1crop.size(2),xup.size(2)))
        xup = self.bn8_out(torch.cat((x1, xup), 1))  # x9 cat x1
        xup = F.relu(self.conv9_3(F.relu(self.conv9_2(F.relu(self.conv9_1(xup))))))  # x9out

        return F.softsign(self.bn9(xup))



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


unet = UNet().cuda()

from os.path import exists, join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale
import torch.utils.data as data
from os import listdir
from PIL import Image


def bsd500(dest="/dir/to/dataset"):#自行修改路径！！

    if not exists(dest):
        print("dataset not exist ")
    return dest


def input_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor()
    ])


def get_training_set(size, target_mode='seg', colordim=1):
    root_dir = bsd500()
    train_dir = join(root_dir, "train")
    return DatasetFromFolder(train_dir,target_mode,colordim,
                             input_transform=input_transform(size),
                             target_transform=input_transform(size))


def get_test_set(size, target_mode='seg', colordim=1):
    root_dir = bsd500()
    test_dir = join(root_dir, "test")
    return DatasetFromFolder(test_dir,target_mode,colordim,
                             input_transform=input_transform(size),
                             target_transform=input_transform(size))




def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath,colordim):
    if colordim==1:
        img = Image.open(filepath).convert('L')
    else:
        img = Image.open(filepath).convert('RGB')
    #y, _, _ = img.split()
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, target_mode, colordim, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [x for x in listdir( join(image_dir,'data') ) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.image_dir = image_dir
        self.target_mode = target_mode
        self.colordim = colordim

    def __getitem__(self, index):


        input = load_img(join(self.image_dir,'data',self.image_filenames[index]),self.colordim)
        if self.target_mode=='seg':
            target = load_img(join(self.image_dir,'seg',self.image_filenames[index]),1)
        else:
            target = load_img(join(self.image_dir,'bon',self.image_filenames[index]),1)


        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

from __future__ import print_function
from math import log10
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from unet import UNet
from BSDDataLoader import get_training_set,get_test_set
import torchvision


# Training settings
class option:
    def __init__(self):
        self.cuda = True #use cuda?
        self.batchSize = 4 #training batch size
        self.testBatchSize = 4 #testing batch size
        self.nEpochs = 140 #umber of epochs to train for
        self.lr = 0.001 #Learning Rate. Default=0.01
        self.threads = 4 #number of threads for data loader to use
        self.seed = 123 #random seed to use. Default=123
        self.size = 428
        self.remsize = 20
        self.colordim = 1
        self.target_mode = 'bon'
        self.pretrain_net = "/home/wcd/PytorchProject/Unet/unetdata/checkpoint/model_epoch_140.pth"

def map01(tensor,eps=1e-5):
    #input/output:tensor
    max = np.max(tensor.numpy(), axis=(1,2,3), keepdims=True)
    min = np.min(tensor.numpy(), axis=(1,2,3), keepdims=True)
    if (max-min).any():
        return torch.from_numpy( (tensor.numpy() - min) / (max-min + eps) )
    else:
        return torch.from_numpy( (tensor.numpy() - min) / (max-min) )


def sizeIsValid(size):
    for i in range(4):
        size -= 4
        if size%2:
            return 0
        else:
            size /= 2
    for i in range(4):
        size -= 4
        size *= 2
    return size-4



opt = option()
target_size = sizeIsValid(opt.size)
print("outputsize is: "+str(target_size))
if not target_size:
    raise  Exception("input size invalid")
target_gap = (opt.size - target_size)//2
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.size + opt.remsize, target_mode=opt.target_mode, colordim=opt.colordim)
test_set = get_test_set(opt.size, target_mode=opt.target_mode, colordim=opt.colordim)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building unet')
unet = UNet(opt.colordim)


criterion = nn.MSELoss()
if cuda:
    unet = unet.cuda()
    criterion = criterion.cuda()

pretrained = True
if pretrained:
    unet.load_state_dict(torch.load(opt.pretrain_net))

optimizer = optim.SGD(unet.parameters(), lr=opt.lr)
print('===> Training unet')

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        randH = random.randint(0, opt.remsize)
        randW = random.randint(0, opt.remsize)
        input = Variable(batch[0][:, :, randH:randH + opt.size, randW:randW + opt.size])
        target = Variable(batch[1][:, :,randH + target_gap:randH + target_gap + target_size,randW + target_gap:randW + target_gap + target_size])
        #target =target.squeeze(1)
        #print(target.data.size())
        if cuda:
            input = input.cuda()
            target = target.cuda()
        input = unet(input)
        #print(input.data.size())
        loss = criterion( input, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        if iteration%10 is 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
    imgout = input.data/2 +1
    torchvision.utils.save_image(imgout,"/home/wcd/PytorchProject/Unet/unetdata/checkpoint/epch_"+str(epoch)+'.jpg')
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    totalloss = 0
    for batch in testing_data_loader:
        input = Variable(batch[0],volatile=True)
        target = Variable(batch[1][:, :,
                          target_gap:target_gap + target_size,
                          target_gap:target_gap + target_size],
                          volatile=True)
        #target =target.long().squeeze(1)
        if cuda:
            input = input.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        prediction = unet(input)
        loss = criterion(prediction, target)
        totalloss += loss.data[0]
    print("===> Avg. test loss: {:.4f} dB".format(totalloss / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "/home/wcd/PytorchProject/Unet/unetdata/checkpoint/model_epoch_{}.pth".format(epoch)
    torch.save(unet.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(141, 141+opt.nEpochs + 1):
    train(epoch)
    if epoch%10 is 0:
        checkpoint(epoch)
    test()
checkpoint(epoch)