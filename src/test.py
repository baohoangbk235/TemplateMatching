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
from sklearn.cluster import KMeans
from collections import Counter
import cv2 
import matplotlib.pyplot as plt 
import os
import pickle 

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

def predict(nn, X_train, y_train, X_test):
    nbrs = KNeighborsClassifier(n_neighbors=nn).fit(X_train, y_train)
    # distances, indices = nbrs.kneighbors(X_test)
    # all_top5 = y_train[indices]
    # results = []
    # for top5 in all_top5:
    #     result = most_frequent(list(top5))
    #     results.append(result)
    results = nbrs.predict(X_test)
    return results

def KMean_predict(X_train, y_train, X_test):
    kmeans = KMeans(n_clusters=8, random_state=0).fit(X_train, y_train)
    results = kmeans.predict(X_test)
    return results

def show(f, positive=False):
    img = cv2.imread(f[0])
    plt.figure(figsize=(20,20))
    plt.xlabel("{} - Ground truth: {} - Predict: {}".format(os.path.basename(f[0]), int2label[f[2]], int2label[f[1]]), fontsize=18)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if positive:
        plt.savefig(f'results/trues/{os.path.basename(f[0])}')
    else:
        plt.savefig(f'results/falses/{os.path.basename(f[0])}')
    
def test():
    with open('train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    train_embs, train_labels, train_paths = train_data["X"], train_data["y"], train_data["path"]
    with open('test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    test_embs, test_labels, test_paths = test_data["X"], test_data["y"], test_data["path"]
    for i in range(1,2):    
        results = predict(i, train_embs, train_labels, test_embs)
        count = 0
        samples = []
        false = []
        for i, result in enumerate(results):
            if result == test_labels[i]:
                pos = True 
                count += 1
            else:
                pos = False
            show([test_paths[i], test_labels[i], result], positive=pos)
        print(count/len(results))


if __name__ == "__main__":
    test()