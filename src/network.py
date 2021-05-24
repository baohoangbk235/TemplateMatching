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
import torch.nn.functional as F
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
        layers = [nn.Linear(num_ftrs, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes)]
        self.model.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        x =  self.model(x)
        return F.normalize(x, p =2, dim = 1)

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

class HardBatchTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(HardBatchTripletLoss, self).__init__()
        self.margin = margin

    def _pairwise_distance(self, embeddings, squared=False):
        dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))	
        square_norm = torch.diagonal(dot_product, offset=0)
        distance = torch.unsqueeze(square_norm, 0) - 2 * dot_product + torch.unsqueeze(square_norm, 1)
        distance = torch.maximum(distance, torch.zeros(distance.shape))
        if not squared:
            mask = torch.tensor(torch.equal(distance, torch.zeros(distance.shape)), dtype=torch.float64)
            distance = distance + mask * 1e-6
            distance = torch.sqrt(distance)
            distance = distance * (1.0 - mask)

        return distance

    def _get_anchor_positive_triplet_mask(self, labels):
        labels_tensor = torch.tensor(labels)
        labels_matrix = torch.stack(len(labels)*[labels_tensor], dim=1)
        mask = (labels_matrix == torch.transpose(labels_matrix, 0, 1)).float()
        mask = mask.fill_diagonal_(0)
        return mask 

    def _get_anchor_negative_triplet_mask(self, labels):
        labels_tensor = torch.tensor(labels)
        labels_matrix = torch.stack(len(labels)*[labels_tensor], dim=1)
        mask = (labels_matrix != torch.transpose(labels_matrix, 0, 1)).float()
        mask = mask.fill_diagonal_(0)
        return mask 

    def forward(self, labels, embeddings, squared=False):
        pairwise_dist  = self._pairwise_distance(embeddings)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)
        anchor_positive_dist = torch.multiply(mask_anchor_positive, pairwise_dist)
        hardest_positive_dist = torch.max(anchor_positive_dist, axis=1, keepdims=True)[0]

        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)
        max_anchor_negative_dist = torch.max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist[0] * (1.0 - mask_anchor_negative)
        hardest_negative_dist = torch.min(anchor_negative_dist, axis=1, keepdims=True)[0]
        triplet_loss = torch.maximum(hardest_positive_dist - hardest_negative_dist + self.margin, torch.zeros(pairwise_dist.shape))
        return triplet_loss.mean()
    
   

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



if __name__ == "__main__":
    net = Network(20)
