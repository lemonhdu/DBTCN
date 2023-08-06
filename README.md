# DBTCN
Pytorch implementation for DBTCN. Two datasets, namely 50Salads and GTEA are tested in the experiment.

# Usage
## Requirements
1. Pytorch >= 1.12.1
2. python >= 3.8
3. tqdm
4. opencv >= 4.7

## Dataset Preparation
1. Download the dataset from [here](https://zenodo.org/record/3625992#.Xiv9jGhKhPY). The repository contains the extracted 3D features, ground truth files, and mapping files.
2. Download our cropped local images from here.
3. We have re-annotated the action labels of the GTEA dataset, which is based on the original annotations of GTEA with 71 action classifications. The new annotation files can be downloaded from [here](https://drive.google.com/file/d/16XJeR-_itj_Tmq_qFn3bSQkirClXvfk2/view?usp=drive_link).

## Network train

### Local image feature
1. Train the local images with ResNet50. Split the training set and testing set according to the official splitting manners of 50Salads and GTEA. 
2. Using the trained ResNet50 corresponding to each split to extract the local image features.

### Action segmentation network
1. python train.py -- action train -- dataset 50salads --split 1 --epoch 100
2. python train.py -- action predict -- dataset 50salads --split 1 --epoch 100

# Contact
If you have any questions, please do not hesitate to contact us. The contact e-mail is huangxvfeng@hdu.edu.cn.
