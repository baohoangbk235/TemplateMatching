import torch
import torch.nn as nn
from torchvision import  models, transforms
from dataset import EmbeddingDataset, TemplateDataset, collate_fn, collate_test, get_train_transforms
import yaml 
from utils import load_checkpoint, get_labels
import argparse
import re 
from torch.optim import lr_scheduler
import torch.optim as optim
import torch 
import torch.nn as nn 
from torch.utils.data import  DataLoader
from tqdm import tqdm 
import numpy as np 
import pickle 

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self.num_classes = num_classes
        self.model = models.vgg16(pretrained=True)
        num_ftrs = self.model.classifier[0].in_features
        layers = [nn.Linear(num_ftrs, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5, inplace=False), nn.Linear(1024, 256), nn.ReLU(inplace=True), nn.Dropout(0.5, inplace=False), nn.Linear(256, 64)]
        self.model.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor, pos, neg):
        pos_distance = self.calc_euclidean(anchor, pos)
        neg_distance = self.calc_euclidean(anchor, neg)
        losses = torch.relu(pos_distance - neg_distance + self.margin)
        return losses.mean()

class Classifier(nn.Module):
    def __init__(self, nb_classes):
        super().__init__()
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, nb_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.softmax(x)
        return x

def save_trained_embedding(model, optimizer, scheduler, CONFIG, test=True):
    if not test:
        data_path = CONFIG['train_data']
    else:
        data_path = CONFIG['test_data']
    
    print("[INFO] Loading model to get embedding ...")
    label2int, int2label = get_labels(CONFIG['labels'])
    model, optimizer, scheduler, loss_history = load_checkpoint(CONFIG['model_path'], model, optimizer, scheduler)
    model.eval()
    model.to(CONFIG["device"])
    traindataset = TemplateDataset(data_path, label2int, transforms=get_train_transforms(), test=True, show=True) 
    trainloader = DataLoader(traindataset, batch_size=CONFIG["batch_size"], shuffle=True)

    list_embs = []
    list_labels = []
    list_paths = []

    for i, (images, labels, paths) in enumerate(trainloader):
        anchors = images[0]
        anchors = anchors.to(CONFIG["device"])
        anchor_emb = model(anchors)  
        list_embs.extend(list(anchor_emb.cpu().detach().numpy()))
        list_labels.extend(list(labels.cpu().detach().numpy()))
        list_paths.extend(paths)
        del anchor_emb
        
    print("[INFO] Get embeddings done!")
    data = {
        "X": np.array(list_embs),
        "y": np.array(list_labels),
        "path": list_paths
    }
    filename = "test.pkl" if test else "train.pkl"
    with open(filename, "wb") as f:
        pickle.dump(data, f)

    

if __name__ == "__main__":
    net = Network(20)
