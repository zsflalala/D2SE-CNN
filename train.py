from ast import arg, parse
from email import parser
import numpy as np
import megengine as mge
import megengine.module as M
import megengine.functional as F
from megengine.utils.module_stats import module_stats
from functools import partial
import math
import megengine.data as data
from megengine.data.dataset import Dataset
from megengine.data.sampler import SequentialSampler
import cv2
import megengine.data.transform as T
import megengine.optimizer as optim
import megengine.autodiff as autodiff
import os
import matplotlib.pyplot as plt
import time
import copy
from skimage.metrics import  peak_signal_noise_ratio,structural_similarity,mean_squared_error
import argparse
from loaddata import *
from Model import *

parser = argparse.ArgumentParser(description="B2SE-CNN training")
parser.add_argument('--datapath','--dp',type=str,default='./data/NWPUVHR',
                    help='choose one root datapath(default="./data/NWPUVHR")')
parser.add_argument('--noisedatapath','--ndp',type=str,default='noise_0.8',
                    help='choose one noise input(default="noise_0.8")')
parser.add_argument('--gtdatapath','--gdp',type=str,default='mask_gray',
                    help='choose one Corresponding ground truth(default="mask_gray")')
parser.add_argument('--savefilename','--s',type=str,default='best.pth.tar',
                    help="Your save file name(default='best.pth.tar')")
parser.add_argument('--batchsize','--b',type=int,default=8)
parser.add_argument('--cropsize','--cz',type=int,default=256,help="Image crop size(default=256)")
parser.add_argument('--resume','--r',type=str,default='',help='resume_root (default:"")')
parser.add_argument('--learning_rate', '--lr', default=0.0002, type=float,
                    help='initial learning rate')
parser.add_argument('--epoch', '--e', default=100, type=int)

global args
args = parser.parse_args()

# 总方差损失计算
def total_variation(image_in):
    tv_h = F.sum(F.abs(image_in[:, :-1] - image_in[:, 1:]))
    tv_w = F.sum(F.abs(image_in[:-1, :] - image_in[1:, :]))
    tv_loss = tv_h + tv_w
    return tv_loss

# Tv_Loss Tv损失函数
def TV_loss(im_batch, weight):
    TV_L = 0.0
    for tv_idx in range(len(im_batch)):
        TV_L = TV_L + total_variation(im_batch[tv_idx, 0, :, :])
    TV_L = TV_L / len(im_batch)
    return weight * TV_L


# save the model parameters
def save_checkpoint(state, filename='best.pth.tar'):
    mge.save(state, filename)

# 展示输出函数 输入的图片维度 （b, c , h, w）
def show_out(img_out):
    img_out = F.clip(img_out[0].transpose(1,2,0),0,1)
    plt.imshow(img_out,cmap=plt.get_cmap('gray'))
    plt.show()

# 可视化epoch&loss曲线图
def show_loss(epoch_list,loss_list):
    plt.plot(epoch_list, loss_list)
    plt.title('Traning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# 定义批处理大小
batch_size = args.batchsize
# 定义裁剪图片的大小 256*256
crop_size = args.cropsize

#这里是数据集文件夹的路径名
train_dataset = SAR_Data(args.datapath, args.noisedatapath,args.gtdatapath,crop_size, training_set="train") #里面是图片的tensor数据,(3,256,236)
train_sampler = data.RandomSampler(train_dataset,batch_size=batch_size)
train_dataloader = data.DataLoader(train_dataset,train_sampler,num_workers=6)

val_dataset = SAR_Data(args.datapath, args.noisedatapath,args.gtdatapath,crop_size, training_set="val")
val_sampler = data.RandomSampler(val_dataset,batch_size=batch_size)
val_dataloader = data.DataLoader(val_dataset,val_sampler,num_workers=6)

model = D2SE_CNN()

if args.resume != '':
    Checkpoint=mge.load(args.resume)
    state_dict = Checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print('parameter loaded successfully!!')
    print("best_epoch : ",Checkpoint["best_epoch"])
    print("best_loss : ",Checkpoint["best_loss"])

# 定义损失函数
criterion = F.nn.l1_loss
# criterion = F.nn.square_loss
# 定义学习率
learning_rate = args.learning_rate
# 定义衰减率
weight_decay = 1e-5
# 定义优化器
gm = autodiff.GradManager().attach(model.parameters())
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

# 记录开始时间
since = time.time()
# 运行epoch数
num_epochs = args.epoch
# 设置最好的loss
if args.resume != '':
    best_loss =  Checkpoint["best_loss"]
else:
    best_loss = 1

# 设置loss列表和epoch列表，图示损失函数图
loss_list = []
epoch_list = []

# 训练过程
for epoch in range(num_epochs):
    model.train()  # 设置训练模式
    running_loss = 0.0 # 临时loss，最后叠加算出总和、平均loss
    for (img, mask) in train_dataloader:
        # img.shape (16, 1, 256, 256) 即训练的输入形状
        img = mge.Tensor(img).transpose(0,3,1,2)
        mask = mge.Tensor(mask).transpose(0,3,1,2)
        with gm:
            img_out = model(img)
            # print("img_out.shape",img_out.shape) #img_out.shape (16, 3, 256, 256)
            # print("mask.shape",mask.shape) #mask.shape (16, 3, 256, 256)
            loss = criterion(img_out,mask)
            loss = loss + TV_loss(img_out, 0.000005)
            gm.backward(loss)
            optimizer.step().clear_grad()
        running_loss += loss.item()
    if epoch % 10 == 0 and epoch:
        show_out(img_out)
    epoch_loss = running_loss / len(train_dataset)
    epoch_list.append(epoch)
    loss_list.append(epoch_loss)
    print('Epoch {}/{} ,Train Loss: {:.8f}'.format(epoch, num_epochs - 1, epoch_loss))
    if epoch % 20 == 0 and epoch != 0:
        model.eval()
        running_loss = 0
        for (img, mask) in val_dataloader:
            img = mge.Tensor(img).transpose(0, 3, 1, 2)
            mask = mge.Tensor(mask).transpose(0, 3, 1, 2)
            img_out = model(img)
            loss = criterion(img_out,mask)
#             loss = loss + TV_loss(img_out, 0.000005)
            running_loss += loss.item()
        epoch_loss = running_loss / len(val_dataset)
        print('Epoch {}/{} ,Val Loss (MSE): {:.8f}'.format(epoch, num_epochs - 1, epoch_loss))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            print("best_epoch: ", best_epoch, "best_epoch_loss: ", epoch_loss)
            # 如果产生最好的loss值就存储
            save_checkpoint({"best_epoch":best_epoch,"best_epoch_loss":epoch_loss,"state_dict":model.state_dict()},
                            args.savefilename)

time_elapsed = time.time() - since
# 显示loss函数曲线图
show_loss(epoch_list,loss_list)
# 打印训练时间
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# 打印最好的测试loss损失
print('Best val loss: {:6f}'.format(best_loss))


