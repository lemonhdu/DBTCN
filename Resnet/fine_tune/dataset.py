import os
import random

from torchvision import transforms
import numpy as np

from params import *
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, mode, args):
        super(ImageDataset, self).__init__()
        self.mode = mode
        self.args = args
        self.split = args.split
        self.image_paths, self.labels = self.get_annotation(mode, self.split)

    def get_annotation(self, mode, split):
        if mode is "train":
            index_dir = r"../resources/gtea/train_idx_"+str(split)+".npy"
        if mode is "valid":
            index_dir = r"../resources/gtea/valid_idx_"+str(split)+".npy"
        elif mode is "test":
            index_dir = r"../resources/gtea/test_idx_"+str(split)+".npy"

        annotation_content = np.load(index_dir)
        image_paths = []
        image_labels = []

        for item in annotation_content:
            video_name = item.split(" ")[0]
            image_num = int(item.split(" ")[1])
            action_label = action_dict[item.split(" ")[2]]

            image_paths.append(os.path.join(video_name, "{:05d}".format(image_num) + ".jpg"))
            image_labels.append(action_label)

        return image_paths, image_labels

    def get_image(self, ix):
        image_path = self.image_paths[ix]
        image_path_complete = os.path.join(videos_dir, image_path)
        image = Image.open(image_path_complete)
        resize = transforms.Resize((224, 224))
        # image = image.convert("RGB")
        image = resize(image)
        # crop_image = transforms.RandomCrop((224, 244))
        # image = crop_image(image)
        # image = np.load(image_path_complete)
        # salads 50salads_local_2d_images
        # normalize = transforms.Normalize(mean=[0.4346, 0.3907, 0.3228], std=[0.1821, 0.1897, 0.1940])
        # gtea gtea_local_2d_images
        normalize = transforms.Normalize([0.5261, 0.3649, 0.1238], [0.1246, 0.1418, 0.0887])
        to_Tensor = transforms.ToTensor()
        image = to_Tensor(image)
        image = normalize(image)
        return image

    def __getitem__(self, ix):
        data = {'image': self.get_image(ix), 'label': self.labels[ix]}
        return data

    def __len__(self):
        image_num = len(self.labels)
        return image_num
