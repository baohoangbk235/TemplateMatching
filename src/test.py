from network import Classifier, Network
from dataset import EmbeddingDataset, TemplateDataset, collate_fn, collate_test, get_train_transforms
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
parser.add_argument("-f", "--first_time", help="first time run testing", type=bool, default=False)
parser.add_argument("--classifier", help="classifier", type=str, default="knn")

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

def predict(X_train, y_train, X_test):
    nbrs = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    # distances, indices = nbrs.kneighbors(X_test)
    # all_top5 = y_train[indices]
    # results = []
    # for top5 in all_top5:
    #     result = most_frequent(list(top5))
    #     results.append(result)
    results = nbrs.predict(X_test)
    return results

def MSEpredict(X_train, y_train, X_test):
    mat_train = np.stack([X_train for _ in range(X_test.shape[0])], 0)
    mat_test = np.stack([X_test for _ in range(X_train.shape[0])], 1)
    distance = np.power((mat_train - mat_test), 2).sum(axis=2)
    nearest = np.argmin(distance, axis=1)
    return y_train[nearest]

def KMean_predict(X_train, y_train, X_test):
    kmeans = KMeans(n_clusters=8, random_state=0).fit(X_train, y_train)
    results = kmeans.predict(X_test)
    return results

def FCpredict(X_test, y_test):
    device = args.device
    config = torch.load("/content/drive/MyDrive/DATN/weights/classifier/best.pth")
    net = Classifier(9)
    net.load_state_dict(config["model_state_dict"])
    net.eval()
    net.to(device)

    dataset = EmbeddingDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_test)
    correct = 0
    total = 0
    for i, (X, yhat) in enumerate(loader):
        X = torch.stack([x for x in X], dim=0)
        
        yhat = torch.tensor(yhat, dtype=torch.long)
        yhat = torch.stack([y for y in yhat], dim=0)
        
        X = X.to(device)
        yhat = yhat.to(device)

        y = net(X)

        _, predicted = torch.max(y.data, 1)
        correct += (predicted  == yhat).sum().item()
        total += yhat.size(0)
        
    print(correct/total)


def show(f, positive=False):
    img = cv2.imread(f[0])
    plt.figure(figsize=(20,20))
    plt.xlabel("{} - Ground truth: {} - Predict: {}".format(os.path.basename(f[0]), int2label[f[1]], int2label[f[2]]), fontsize=25)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if positive:
        plt.savefig(f'results/trues/{os.path.basename(f[0])}')
    else:
        plt.savefig(f'results/falses/{os.path.basename(f[0])}')
    

def test():
    import shutil
    data = get_trained_embeddings()
    if os.path.exists("results/falses/"):
        shutil.rmtree("results/falses/")
    if os.path.exists("results/trues/"):
        shutil.rmtree("results/trues/")
    os.makedirs("results/falses/", exist_ok =True)
    os.makedirs("results/trues/", exist_ok =True)

    if args.classifier == "fcn":
        FCpredict(data["test"][0], data["test"][1])
    else:
        results = predict(data["train"][0], data["train"][1], data["test"][0])
        print(np.sum(results==data["test"][1])/results.shape[0])

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
    os.remove(filename)
    with open(filename, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    if args.first_time:
        model = Network(9)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_change_lr, gamma=0.5)

        save_trained_embedding(model, optimizer, scheduler, CONFIG, test=True)
        save_trained_embedding(model, optimizer, scheduler, CONFIG, test=False)
    test()