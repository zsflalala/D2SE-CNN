import os
from megengine.data.dataset import Dataset
import megengine.data.transform as T
import cv2
import numpy as np

# 自定义训练数据集
class SAR_Data(Dataset):
    def __init__(self, dataset_path, noisedatapath,gtdatapath,crop_size, training_set) -> None:
        self.dataset_path = dataset_path
        # 选择噪声图像路径
        self.noisy_path = os.path.join(dataset_path, noisedatapath)
        # 选择噪声图对应的真值图
        self.clean_path = os.path.join(dataset_path, gtdatapath)
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
            # 输出噪声图和相应的真值图列表，看是否对应
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

    # 返回图片个数
    def __len__(self):
        return len(self.images_noisy_list)

    # 获取图片
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

        # 灰度图的维度扩展
        image = np.expand_dims(image, 2)
        mask = np.expand_dims(mask, 2)
        
        # 噪声图重新裁剪
        image = T.Resize((int(self.crop * 1.25), int(self.crop * 1.25))).apply(image) 
        image = image.astype(np.float32) / 255

        # 真值图重新裁剪
        mask = T.Resize((int(self.crop * 1.25), int(self.crop * 1.25))).apply(mask) 
        mask = mask.astype(np.float32) / 255
        # 中心裁剪
        image,mask = T.CenterCrop((self.crop)).apply(image),T.CenterCrop((self.crop)).apply(mask)

        # 噪声图正则预处理
        image = T.Normalize(mean=[0.456], std=[0.224]).apply(image)
        return image,mask
