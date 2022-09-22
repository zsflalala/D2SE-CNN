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

# D2SE_CNN-Net
class D2SE_CNN(M.Module):
    def __init__(self):
        super(D2SE_CNN, self).__init__()
        # self.lamb = lambda x:( x + M.init.fill_(x,1e-7))
        self.Conv_BN_ReLU1 = M.Sequential(
            M.Conv2d(64, 64, 3, dilation=1, padding=1, stride=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU()
        )
        self.Conv_BN_ReLU2 = M.Sequential(
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU()
        )
        self.Conv_BN_ReLU3 = M.Sequential(
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU()
        )
        self.Conv_BN_ReLU = M.Sequential(
            M.Conv2d(64, 64, 3, dilation=1, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=2, padding=2),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=2, padding=2),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=2, padding=2),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=3, padding=3),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=3, padding=3),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=2, padding=2),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=2, padding=2),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=2, padding=2),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=1, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=1, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=1, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=1, padding=1),
            M.BatchNorm2d(64),
            M.ReLU())
        self.dia = M.Sequential(
            M.Conv2d(1, 4, 3, dilation=65, padding=1)

        )
        self.Conv_ReLU_L1 = M.Sequential(
            M.Conv2d(4, 64, 3, padding=1),
            M.ReLU()
        )
        self.Conv_ReLU_L8 = M.Sequential(
            M.Conv2d(64, 4, 3, padding=1)
        )
        self.ConvTrans = M.ConvTranspose2d(4, 4, 4, stride=2, padding=1)
        self.proj = M.Conv2d(1, 1, 5, padding=2, stride=1)
        self.avg_pool = M.AdaptiveAvgPool2d(1)
        self.fc = M.Sequential(
            M.Linear(64, 4, bias=False),
            M.ReLU(),
            M.Linear(4, 64, bias=False),
            M.Sigmoid()
        )
        self.apply(self._init_weights)


    # 初始化各层参数
    def _init_weights(self, m):
        if isinstance(m, M.Conv2d):
            # print("Conv2d")
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            M.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            # M.init.msra_normal_(m.weight,mode='fan_in')
            if m.bias is not None:
                M.init.zeros_(m.bias)
        if isinstance(m, M.BatchNorm2d):
            # print("Batch")
            M.init.msra_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                M.init.zeros_(m.bias)
        if isinstance(m, M.LayerNorm):
            # print("layerNorm")
            M.init.zeros_(m.bias)
            M.init.ones_(m.weight)
        if isinstance(m, M.Linear):
            # print("M.linear")
            M.init.normal_(m.weight, std=.02)
            if isinstance(m, M.Linear) and m.bias is not None:
                M.init.zeros_(m.bias)


    def forward(self, x):
        n, c, h, w = x.shape
        # 降采样
        x = x.reshape(n, c, h // 2, 2, w // 2, 2).transpose(0, 1, 3, 5, 2, 4)
        x = x.reshape(n, c * 4, h // 2, w // 2)
        x = self.Conv_ReLU_L1(x)
        # SE 通道注意力
        b, c2, _, _ = x.shape
        y = F.reshape(self.avg_pool(x), (b, c2))  # squeeze操作
        y = F.reshape(self.fc(y), (b, c2, 1, 1))  # FC获取通道注意力权重，是具有全局信息的
        x = x * F.broadcast_to(y, (x.shape))  # 注意力作用每一个通道上
        x = self.Conv_BN_ReLU1(x)
        b, c2, _, _ = x.shape
        y = F.reshape(self.avg_pool(x), (b, c2))  # squeeze操作
        y = F.reshape(self.fc(y), (b, c2, 1, 1))  # FC获取通道注意力权重，是具有全局信息的
        x = x * F.broadcast_to(y, (x.shape))  # 注意力作用每一个通道上
        x = self.Conv_BN_ReLU2(x)
        b, c2, _, _ = x.shape
        y = F.reshape(self.avg_pool(x), (b, c2))  # squeeze操作
        y = F.reshape(self.fc(y), (b, c2, 1, 1))  # FC获取通道注意力权重，是具有全局信息的
        x = x * F.broadcast_to(y, (x.shape))  # 注意力作用每一个通道上
        x = self.Conv_BN_ReLU3(x)
        # x = self.Conv_BN_ReLU(x)
        x = self.Conv_ReLU_L8(x)
        x = x.reshape(n, c, 2, 2, h // 2, w // 2).transpose(0, 1, 4, 2, 5, 3)
        x = x.reshape(n, c, h, w)
        return x

# 自定义训练数据集
class SAR_Data(Dataset):
    def __init__(self, dataset_path, crop_size, training_set) -> None:
        self.dataset_path = dataset_path
        # 选择噪声图像路径
        self.noisy_path = os.path.join(dataset_path, 'noise_0.8')
        # 选择噪声图对应的真值图
        self.clean_path = os.path.join(dataset_path, 'mask_gray')
        self.images_noisy_list = sorted(os.listdir(self.noisy_path))
        self.images_clean_list = sorted(os.listdir(self.clean_path))
        # print(self.images_clean_list)
        self.training_set = training_set
        self.crop = crop_size
        if training_set == "train":  # 60%
            self.images_noisy_list = self.images_noisy_list[0:int(0.6 * len(self.images_noisy_list))]
            if self.images_clean_list[0] == '.DS_Store':
                self.images_clean_list = self.images_clean_list[1:int(0.6 * len(self.images_clean_list)+1)]
            else:
                self.images_clean_list = self.images_clean_list[0:int(0.6 * len(self.images_clean_list))]
            # print(self.images_noisy_list)
            # print(self.images_clean_list)
        elif training_set == "val":  # 20% = 60%->80%
            self.images_noisy_list = self.images_noisy_list[int(0.6 * len(self.images_noisy_list)):int(0.8 * len(self.images_noisy_list))]
            if self.images_clean_list[0] == '.DS_Store':
                self.images_clean_list = self.images_clean_list[int(0.6 * len(self.images_clean_list)+1):int(0.8 * len(self.images_clean_list)+1)]
            else:
                self.images_clean_list = self.images_clean_list[int(0.6 * len(self.images_clean_list)):int(0.8 * len(self.images_clean_list))]
        elif training_set == "test":
            self.images_noisy_list = self.images_noisy_list[int(0.8 * len(self.images_noisy_list)):]
            if self.images_clean_list[0] == '.DS_Store':
                self.images_clean_list = self.images_clean_list[int(0.8 * len(self.images_clean_list)+1):]
            else:
                self.images_clean_list = self.images_clean_list[int(0.8 * len(self.images_clean_list)):]


    def __len__(self):
        return len(self.images_noisy_list)

    def __getitem__(self, idx) -> tuple :
        image_noisy_filename = os.path.join(self.noisy_path,self.images_noisy_list[idx])
        image_clean_filename = os.path.join(self.clean_path,self.images_clean_list[idx])
        # print(image_noisy_filename)
        # print(image_clean_filename)
        image = cv2.imread(image_noisy_filename,0)
        mask = cv2.imread(image_clean_filename,0)
        #取出的数据均是0-256，
#         image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#         mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, 2)
        mask = np.expand_dims(mask, 2)
        
#         print(image)
        image = T.Resize((int(self.crop * 1.25), int(self.crop * 1.25))).apply(image) 
        image = image.astype(np.float32) / 255
#         print(image)
        mask = T.Resize((int(self.crop * 1.25), int(self.crop * 1.25))).apply(mask) 
        mask = mask.astype(np.float32) / 255
        image,mask = T.CenterCrop((self.crop)).apply(image),T.CenterCrop((self.crop)).apply(mask)
        image = T.Normalize(mean=[0.456], std=[0.224]).apply(image)
        return image,mask

# 可视化epoch&loss曲线图
def show_loss(epoch_list,loss_list):
    plt.plot(epoch_list, loss_list)
    plt.title('Traning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

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

# 定义批处理大小
batch_size = 8
# 定义裁剪图片的大小 256*256
crop_size = 256

#这里是数据集文件夹的路径名
train_dataset = SAR_Data("./data/NWPUVHR", crop_size, training_set="train") #里面是图片的tensor数据,(3,256,236)
train_sampler = data.RandomSampler(train_dataset,batch_size=batch_size)
train_dataloader = data.DataLoader(train_dataset,train_sampler,num_workers=15)

val_dataset = SAR_Data("./data/NWPUVHR", crop_size, training_set="val")
val_sampler = data.RandomSampler(val_dataset,batch_size=batch_size)
val_dataloader = data.DataLoader(val_dataset,val_sampler,num_workers=15)

test_dataset = SAR_Data("./data/NWPUVHR", crop_size, training_set="test")
test_sampler = SequentialSampler(test_dataset,batch_size=batch_size)
test_dataloader = data.DataLoader(test_dataset,test_sampler,num_workers=15)

model = D2SE_CNN()
model.load_state_dict(mge.load("./TrainedModel/D2SECNN_0.8N.pth"))

# 定义损失函数
criterion = F.nn.l1_loss
# criterion = F.nn.square_loss
# 定义学习率
learning_rate = 0.0002
# 定义衰减率
weight_decay = 1e-5
# 定义优化器
gm = autodiff.GradManager().attach(model.parameters())
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
print("Loading Pretrained Model is OK")

# 展示输出函数 输入的图片维度 （b, c , h, w）
def show_out(img_out):
    img_out = F.clip(img_out[0].transpose(1,2,0),0,1)
    plt.imshow(img_out,cmap=plt.get_cmap('gray'))
    plt.show()

# 记录开始时间
since = time.time()
# 运行epoch数
num_epochs = 1001
# 设置最好的loss
best_loss =  0.006392
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
            fout = open('./data/best_epoch.pth', 'wb')
            mge.save(model.state_dict(), fout)
            fout.close()

time_elapsed = time.time() - since
# 显示loss函数曲线图
# show_loss(epoch_list,loss_list)
# 打印训练时间
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# 打印最好的测试loss损失
print('Best val loss: {:6f}'.format(best_loss))


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

# 分别定义原图片PSNR、SSIM、MSE在训练图片和模型输出图片列表，以便图示
PSNR_img,PSNR_imgout = [],[]
SSIM_img,SSIM_imgout = [],[]
MSE_img,MSE_imgout = [],[]
# 生成图片epoch列表
pic_list = np.arange(0,len(test_dataset),1)

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

# 加载本次最好的模型
model.load_state_dict(mge.load("./data/best_epoch.pth"))

# 切换验证模式
model.eval()

# 临时计算测试集中总loss和平均loss
running_loss = 0.0
# 测试过程
for (img, mask) in test_dataloader:
    img = mge.Tensor(img).transpose(0, 3, 1, 2)
    mask = mge.Tensor(mask).transpose(0, 3, 1, 2)
    img_out = model(img)
    img = img*0.224+0.446
    # show_pic(img,img_out,mask)
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