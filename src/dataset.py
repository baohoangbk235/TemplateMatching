import os 
import matplotlib.pyplot as plt
import cv2 
import random 
import glob 
from torch.utils.data import Dataset
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2
import torch

class TemplateDataset(Dataset):
    def __init__(self, root_dir, label2int, transforms=None, test=False, show=False):
        self.root_dir = root_dir 
        self.transforms = transforms 
        self.list_images = []
        self.test = test
        self.show = show
        self.label2int = label2int

        if not os.path.exists(root_dir):
            exit()
        
        self.labels = os.listdir(self.root_dir)
        self.triplets = []
        self.triplets_labels = []
        print("[INFO] Processing dataset ...")
        for label in self.labels: 
            list_paths = glob.glob(os.path.join(self.root_dir, label, "*.jpg"))
            for i, anchor in enumerate(list_paths):
                if not test:
                    neg_label = random.choice([neg_ for neg_ in self.labels if neg_ != label])
                    if len(list_paths) < 2:
                        break
                    positive = random.choice([p for p in list_paths if p != anchor])
                    negative = random.choice(glob.glob(os.path.join(self.root_dir, neg_label, "*.jpg")))
                    self.triplets.append([anchor, positive, negative])
                    self.triplets_labels.append([self.label2int[label], self.label2int[label], self.label2int[neg_label]]) 
                else:
                    self.triplets.append([anchor])
                    self.triplets_labels.append([self.label2int[label]]) 

    def __len__(self):
        return len(self.triplets)
    
    def get_img(self, path):
        path = path.replace(os.sep, '/')
        img = cv2.imread(path)
#         print(img.shape)
        return img 
    
    def show_triplet(self, images):
        if len(images) == 3:
            titles = ["Anchor", "Positive", "Negative"]
            plt.figure(figsize=(12,4))
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.imshow(images[i].astype("uint8"))
                plt.title(titles[i])
            plt.show()
    
    def __getitem__(self, index: int):
        triplets = self.triplets[index]
        labels = self.triplets_labels[index]
        anchor = self.get_img(triplets[0])
        images = [anchor]
        
        if not self.test:
            pos = self.get_img(triplets[1])
            neg = self.get_img(triplets[2])
            images.extend([pos, neg])
            
            if self.show:
                self.show_triplet(images)
        else:
            labels = labels[0]
            triplets = triplets[0]

        if self.transforms:
            images = [self.transforms(image=img)["image"] for img in images]
        return images, labels, triplets

class EmbeddingDataset(Dataset):
    def __init__(self, list_embs, list_labels):
        self.list_embs = list_embs
        self.list_labels = list_labels

    def __getitem__(self, index):
        return torch.Tensor(self.list_embs[index]), self.list_labels[index]
    
    def __len__(self):
        return len(self.list_embs)

class MyDataset(Dataset):
    def __init__(self, root_dir, label2int, transforms=None, test=False):
        self.root_dir = root_dir 
        self.transforms = transforms 
        self.list_images = []
        self.test = test
        self.label2int = label2int

        if not os.path.exists(root_dir):
            exit()

        self.all_img_paths = []
        self.all_labels = []
        print("[INFO] Processing dataset ...")
        for label in self.label2int.keys(): 
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
        image = self.get_img(self.all_img_paths[index])
        label = self.all_labels[index]
        if self.transforms:
            image = self.transforms(image=image)["image"]
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

def collate_test(batch):
    images_list = []
    path_list = []
    for images, paths in batch:
        images_list.append(images)
        path_list.append(paths)

    return images_list, path_list

def get_train_transforms():
    return A.Compose([
        A.Resize(512, 512), 
        A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]) 
