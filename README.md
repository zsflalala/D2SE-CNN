# D2SE-CNN: An improved SAR Image Despeckling Network

Created by ZHANG Yiming，ZHAO Shengfu，ZHENG Xing and WANG Yibo.

### Introduction

Synthetic Aperture Radar (SAR) is a coherent imaging system and as such it strongly suffers from the presence of speckles. 
The presence of speckles degraded the image quality and makes SAR images difficult to interpret, such as image segmentation, detection, and recognition.
Therefore, removing noise in SAR images is of great significance for improving the performance of various computer vision algorithms such as segmentation, detection, and recognition.


## Dependencies
MegEngine1.8
Python 3.7
scipy 1.2.1  
Pillow 6.0.0  
scikit-image 0.17.2  
scikit-learn 0.22.1

## Data Preparation

#### BSD500 & NWPUVHR-10

You may download the dataset from [BaiduNetDisk](https://pan.baidu.com/s/1fhM6EWaT6MZ5wc0kkupYCw?pwd=D2SE) （ShareCode：D2SE）, and unzip it to the ./data folder. You will have the following directory structure:
```
B2SE-CNN_dataset
|_ BSD500_NEW
|   |_ noise_1.0
|   |_ clean_gray
|   |_ noise_1.2
|   |_ noise_0.8
|_ NWPU VHR-10 NEW
|   |_ noise_1.0
|   |_ clean_gray
|   |_ noise_1.2
|   |_ noise_0.8
```

## Training

For training B2SE-CNN on BSD500 or NWPUVHR-10 training dataset, you can run:

```
python train.py
or
python train.py --datapath ./data/NWPUVHR --noisedatapath noise_0.8 --gtdatapath mask_gray --epoch 1000 --savefilename best.pth.tar
```

You can download our trained model from [Baidu Netdisk](https://pan.baidu.com/s/1UuH194wLiVPf_TC291HrnA?pwd=D2SE) (Code: D2SE).

## Evaluation

For testing BS-Net on BSD500 or NWPUVHR-10 testing dataset, you can run:
```
python test.py
or
python train.py --datapath ./data/NWPUVHR --noisedatapath noise_0.8 --gtdatapath mask_gray --loadpath best.pth.tar
```