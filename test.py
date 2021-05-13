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
args = parser.parse_args()

CONFIG = get_config(args.config)
CONFIG['model_path'] = args.model_path

label2int, int2label = get_labels(CONFIG['labels'])

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def predict(list_embs, list_labels):
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(list_embs)
    distances, indices = nbrs.kneighbors(list_embs)
    all_top5 = np.array(list_labels)[indices]
    results = []
    for i, top5 in enumerate(all_top5):
        result = most_frequent(top5)
        results.append(result)
    
    return results

def show(f):
    img = cv2.imread(f[0])
    plt.figure(figsize=(20,20))
    plt.xlabel("{} - Predict: {}, Ground truth: {}".format(os.path.basename(f[0]),int2label[f[1]], int2label[f[2]]), fontsize=18)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.savefig(f'results/falses/{f[0]}')
    
def test():
    model = Network(20)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=CONFIG["step_change_lr"], gamma=0.5)
    list_embs, list_labels, list_paths = get_trained_embedding(model, optimizer, scheduler, CONFIG)
    list_labels = [int(label.cpu().detach().numpy()) for label in list_labels]
    results = predict(list_embs, list_labels)
    count = 0
    true = []
    false = []
    for i, result in enumerate(results):
        if result == list_labels[i]:
            count += 1
        else:
            false.append([list_paths[i], result, list_labels[i]])
    print(count/len(results))
    for f in false:
        show(f)

if __name__ == "__main__":
    test()