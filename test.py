from ast import arg, parse
from email import parser
from tabnanny import check
import numpy as np
from Model import D2SE_CNN
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

# 定义测试相关参数，包括加载路径（loadpath）、批处理大小（batchsize）、裁剪大小（cropsize）、测试数据集根路径（datapath）
# 噪声图路径（noisedatapath）、真值图路径（gtdatapath）
parser = argparse.ArgumentParser(description="B2SE-CNN testing")
parser.add_argument('--loadpath','--lp',type=str,default='best.pth.tar',
                    help='load your trianed model file(default="best.pth.tar")')
parser.add_argument('--batchsize','--b',type=int,default=8)
parser.add_argument('--cropsize','--cz',type=int,default=256,help="Image crop size(default=256)")
parser.add_argument('--datapath','--dp',type=str,default='./data/NWPUVHR',
                    help='choose one root datapath(default="./data/NWPUVHR")')
parser.add_argument('--noisedatapath','--ndp',type=str,default='noise_0.8',
                    help='choose one noise input(default="noise_0.8")')
parser.add_argument('--gtdatapath','--gdp',type=str,default='mask_gray',
                    help='choose one Corresponding ground truth(default="mask_gray")')
args = parser.parse_args()


# 分别展示训练图片img，模型输出图片img_out，真值mask
def show_pic(img,img_out,mask):
    plt.figure(figsize=(30,30))
    plt.subplot(1,3,1)
    plt.imshow(img[0].transpose(1,2,0),cmap=plt.get_cmap('gray'))
    img_out = F.clip(img_out[0].transpose(1,2,0),0,1)
    plt.subplot(1,3,2)
    plt.imshow(img_out,cmap=plt.get_cmap('gray'))
    plt.subplot(1,3,3)
    plt.imshow(mask[0].transpose(1,2,0),cmap=plt.get_cmap('gray'))
    # plt.savefig("Num5_comp2.jpg")
    plt.show()

# 可视化噪声图与真值图、模型输出图与真值图的曲线图
def show_equality(pic_list,img_list,imgout_list,title):
    plt.plot(pic_list, img_list,label = "noice & ture")
    plt.plot(pic_list, imgout_list,linestyle = "--",label = "pred & true")
    plt.title(title)
    plt.xlabel('Pictures')
    plt.legend()
    plt.show()

# 计算并可视化相关指标
def quality(img,img_out,mask,PSNR_img,SSIM_img,MSE_img,PSNR_imgout,SSIM_imgout,MSE_imgout):
    #shape (H,W,C)
    b ,_,_,_ = mask.shape
    for i in range(b):
        img_ = F.clip(img[i,:,:,:].transpose(1,2,0),0,1).numpy()
        img_out_ = F.clip(img_out[i,:,:,:].transpose(1,2,0),0,1).numpy()
        mask_ = F.clip(mask[i,:,:,:].transpose(1,2,0),0,1).numpy()
        mse = mean_squared_error(img_,mask_)
        ssim = structural_similarity(img_,mask_,channel_axis=2)
        psnr = peak_signal_noise_ratio(img_,mask_)
        PSNR_img.append(psnr)
        SSIM_img.append(ssim)
        MSE_img.append(mse)
        mse2 = mean_squared_error(img_out_,mask_)
        ssim2 = structural_similarity(img_out_,mask_,channel_axis=2)
        psnr2 = peak_signal_noise_ratio(img_out_,mask_)
        PSNR_imgout.append(psnr2)
        SSIM_imgout.append(ssim2)
        MSE_imgout.append(mse2)
    
    avg_psnr_pred = sum(i for i in PSNR_imgout) / len(PSNR_imgout)
    avg_ssim_pred = sum(i for i in SSIM_imgout) / len(SSIM_imgout)
    avg_mse_pred = sum(i for i in MSE_imgout) / len(MSE_imgout)
#     print("Batch PSNR avg : ",avg_psnr_pred)
#     print("Batch SSIM avg : ",avg_ssim_pred)
#     print("Batch MSE  avg : ",avg_mse_pred)



model = D2SE_CNN()
# 加载训练好的模型
checkpoint = mge.load(args.loadpath)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)

# 切换验证模式
model.eval()
# 定义批处理大小
batch_size = args.batchsize
# 定义裁剪图片的大小 256*256
crop_size = args.cropsize
# 定义损失函数
criterion = F.nn.l1_loss
# criterion = F.nn.square_loss

# 建立测试集
test_dataset = SAR_Data(args.datapath, args.noisedatapath,args.gtdatapath,crop_size, training_set="test")
test_sampler = SequentialSampler(test_dataset,batch_size=batch_size)
test_dataloader = data.DataLoader(test_dataset,test_sampler,num_workers=6)

# 分别定义原图片PSNR、SSIM、MSE在训练图片和模型输出图片列表，以便图示
PSNR_img,PSNR_imgout = [],[]
SSIM_img,SSIM_imgout = [],[]
MSE_img,MSE_imgout = [],[]
# 生成图片epoch列表
pic_list = np.arange(0,len(test_dataset),1)

# 临时计算测试集中总loss和平均loss
running_loss = 0.0
# 测试过程
for (img, mask) in test_dataloader:
    img = mge.Tensor(img).transpose(0, 3, 1, 2)
    mask = mge.Tensor(mask).transpose(0, 3, 1, 2)
    img_out = model(img)
    img = img*0.224+0.446
    show_pic(img,img_out,mask)
    quality(img,img_out,mask,PSNR_img,SSIM_img,MSE_img,PSNR_imgout,SSIM_imgout,MSE_imgout)
    loss = criterion(img_out, mask)
    running_loss += loss.item()
epoch_loss = running_loss / len(test_dataset)
print('{} Loss (MSE): {:.6f}'.format("test", epoch_loss))

# 可视化指标图
show_equality(pic_list,PSNR_img,PSNR_imgout,"PSNR")
show_equality(pic_list,SSIM_img,SSIM_imgout,"SSIM")
show_equality(pic_list,MSE_img,MSE_imgout,"MSE")

# 打印测试集指标
print("PSNR avg: ",sum(PSNR_imgout) / len(PSNR_imgout))
print("SSIM avg: ",sum(SSIM_imgout) / len(SSIM_imgout))
print("MSE  avg: ",sum(MSE_imgout) / len(MSE_imgout))