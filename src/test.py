from network import Classifier, Network, get_trained_embedding
from dataset import EmbeddingDataset, TemplateDataset, collate_fn, collate_test, get_train_transforms
import yaml 
from utils import load_checkpoint, get_labels, get_config
import argparse
import re 
from torch.optim import lr_scheduler
import torch.optim as optim
import torch 
import torch.nn as nn 
from torch.utils.data import  DataLoader
from tqdm import tqdm 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from collections import Counter
import cv2 
import matplotlib.pyplot as plt 
import os 

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--model_path", help="path to checkpoint of feature extraction model", type=str)
parser.add_argument("-c", "--config", help="data augmentation", type=str, default="config.yaml")
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=5e-4)
parser.add_argument("-w", "--weight_decay", help="weight decay", type=float, default=1e-6)
parser.add_argument("-st", "--step_change_lr", help="step change learning rate", type=int, default=10)
parser.add_argument("-bs", "--batch_size", help="batch size", type=int, default=8)
parser.add_argument("-ep", "--epochs", help="epochs", type=int, default=30)
parser.add_argument("-l", "--label_path", help="epochs", type=str, default="dataset/labels.txt")
parser.add_argument("-d", "--device", help="device", type=str, default="cuda")
args = parser.parse_args()

CONFIG = get_config(args.config)
CONFIG['model_path'] = args.model_path
CONFIG['device'] = args.device 
CONFIG['labels'] = args.label_path 
CONFIG['batch_size'] = args.batch_size

label2int, int2label = get_labels(CONFIG['labels'])

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def predict(list_embs, list_labels):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(list_embs)
    distances, indices = nbrs.kneighbors(list_embs)
    all_top5 = np.array(list_labels)[indices]
    results = []
    for i, top5 in enumerate(all_top5):
        result = most_frequent(top5)
        results.append(result)
    
    return results

def show(f, result=False):
    img = cv2.imread(f[0])
    plt.figure(figsize=(20,20))
    plt.xlabel("{} - Ground truth: {} - Predict: {}".format(os.path.basename(f[0]), int2label[f[2]], int2label[f[1]]), fontsize=18)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if result:
        plt.savefig(f'results/trues/{os.path.basename(f[0])}')
    else:
        plt.savefig(f'results/falses/{os.path.basename(f[0])}')
    
def test():
    model = Network(20)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_change_lr, gamma=0.5)
    list_embs, list_labels, list_paths = get_trained_embedding(model, optimizer, scheduler, CONFIG)
    list_labels = [int(label.cpu().detach().numpy()) for label in list_labels]
    results = predict(list_embs, list_labels)
    count = 0
    samples = []
    false = []
    for i, result in enumerate(results):
        if result == list_labels[i]:
            positive = True 
            count += 1
        else:
            positive = False
        show([list_paths[i], result, list_labels[i]], result=positive)

    print(count/len(results))

if __name__ == "__main__":
    test()