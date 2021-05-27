import os 
import matplotlib.pyplot as plt
import cv2 
import random 
import glob 
from torch.utils.data import Dataset
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2
import torch

class EmbeddingDataset(Dataset):
    def __init__(self, list_embs, list_labels):
        self.list_embs = list_embs
        self.list_labels = list_labels

    def __getitem__(self, index):
        return torch.Tensor(self.list_embs[index]), self.list_labels[index]
    
    def __len__(self):
        return len(self.list_embs)

class TemplateDataset(Dataset):
    def __init__(self, root_dir, label2int, transforms=None, mode="train"):
        self.root_dir = root_dir 
        self.transforms = transforms 
        self.list_images = []
        self.mode = mode
        self.label2int = label2int

        if not os.path.exists(root_dir):
            exit("Not found {}".format(root_dir))

        self.all_img_paths = []
        self.all_labels = []
        print("[INFO] Processing dataset {}...".format(self.root_dir))
        if mode == "test": 
            self.all_img_paths.extend(glob.glob(os.path.join(self.root_dir, "*.jpg"))) 
        else:
            for i, label in enumerate(self.label2int.keys()): 
                list_paths = glob.glob(os.path.join(self.root_dir, label, "*.jpg"))
                self.all_img_paths.extend(list_paths)
                self.all_labels.extend([self.label2int[label]] * len(list_paths))

    def __len__(self):
        return len(self.all_img_paths)
    
    def get_img(self, path):
        path = path.replace(os.sep, '/')
        img = cv2.imread(path)
        return img 
    
    def __getitem__(self, index: int):
        path = self.all_img_paths[index]
        image = self.get_img(path)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        if self.mode == "test":
            return image, path 
        else:
            label = self.all_labels[index]
            if self.mode == "valid":
                return image, label, path 
            return image, label


def collate_fn(batch):
    images_list = []
    labels_list = []
    image_paths = []
    for images, labels, paths in batch:
        images_list.append(images)
        labels_list.append(labels)
        image_paths.append(paths)
        
    return images_list, labels_list, image_paths

def collate_fn_test(batch):
    images_list = []
    image_paths = []
    for images, paths in batch:
        images_list.append(images)
        image_paths.append(paths)
        
    return images_list, image_paths

def get_train_transforms():
    return A.Compose([
        A.Resize(512, 512), 
        A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]) 
