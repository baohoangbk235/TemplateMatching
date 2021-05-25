from network import Classifier, Network
from dataset import EmbeddingDataset, TemplateDataset, collate_fn, get_train_transforms
import yaml 
from utils import load_checkpoint, get_labels, get_config, get_trained_embeddings
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
from Models.Classifier import KNNClassifier, FCNClassifier
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--model_path", help="path to checkpoint of feature extraction model", type=str)
parser.add_argument("-c", "--config", help="data augmentation", type=str, default="config.yaml")
parser.add_argument("-bs", "--batch_size", help="batch size", type=int, default=8)
parser.add_argument("-l", "--label_path", help="epochs", type=str, default="dataset/labels.txt")
parser.add_argument("-d", "--device", help="device", type=str, default="cuda")
parser.add_argument("-f", "--first_time", help="first time run testing", type=bool, default=False)
parser.add_argument("--classifier", help="classifier", type=str, default="knn")
parser.add_argument("--valid", help="valid or test", type=bool, default=False)
parser.add_argument("--emb_size", help="embedding size", type=int, default=9)


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

def recreateFolder(pathToFold):
    if os.path.exists(pathToFold):
        shutil.rmtree(pathToFold)
    os.mkdir(pathToFold)

def test():
    data = get_trained_embeddings()
    recreateFolder("results/trues/")
    recreateFolder("results/falses/")

    if args.classifier == "fcn":
        pass
    else:
        classifier = KNNClassifier()
        classifier.fit(data["train"][0], data["train"][1])
        predicted = classifier.test(data["test"][0])
        if args.valid:
            acc = classifier.evaluate(data["test"][0], data["test"][1])
            compareToVisualize(predicted, data["test"][1], data["test"][2])
            print(f'Valid Accuracy: {acc}')

def compareToVisualize(predicted, groundTruth, paths):
    for i in range(len(predicted)):
        if (predicted[i] == groundTruth[i]):
            pos = True 
        else:
            pos = False 
        show([paths[i], groundTruth[i], predicted[i]], positive=pos)

def show(f, positive=False):
    img = cv2.imread(f[0])
    plt.figure(figsize=(20,20))
    plt.xlabel("{} - Ground truth: {} - Predict: {}".format(os.path.basename(f[0]), int2label[f[1]], int2label[f[2]]), fontsize=25)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if positive:
        plt.savefig(f'results/trues/{os.path.basename(f[0])}')
    else:
        plt.savefig(f'results/falses/{os.path.basename(f[0])}')

def save_trained_embedding(CONFIG):
    label2int, int2label = get_labels(args.label_path)
    config = torch.load(args.model_path)
    model = Network(args.emb_size)
    model.load_state_dict(config["model_state_dict"])
    model.eval()
    model.to(args.device)

    for test in [True, False]:
        if not test:
            data_path = CONFIG['train_data']
        else:
            data_path = CONFIG['test_data']
        
       
        dataset = TemplateDataset(data_path, label2int, transforms=get_train_transforms(), test=True) 
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

        list_embs = []
        list_labels = []
        list_paths = []

        for i, (images, labels, paths) in enumerate(loader):
            anchors = torch.stack(images, axis=0)
            anchors = anchors.to(args.device)
            anchor_emb = model(anchors)  
            list_embs.extend(list(anchor_emb.cpu().detach().numpy()))
            list_labels.extend(labels)
            list_paths.extend(paths)
            del anchor_emb
            
        data = {
            "X": np.array(list_embs),
            "y": np.array(list_labels),
            "path": list_paths
        }
        filename = "test.pkl" if test else "train.pkl"
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, "wb") as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    if args.first_time:
        save_trained_embedding( CONFIG)
    test()