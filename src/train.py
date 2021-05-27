import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import  DataLoader
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import glob
import random
import cv2
from tqdm import trange, tqdm
import warnings
from datetime import datetime
from network import Network, Identity, TripletLoss, HardBatchTripletLoss
from dataset import TemplateDataset, collate_fn, get_train_transforms
from utils import get_labels, get_config, save_config
from tensorboard_logger import log_value, configure
from shutil import copyfile
from Models.Classifier import KNNClassifier, FCNClassifier
import pickle 
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="data augmentation", type=str, default="config.yaml")
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=5e-4)
parser.add_argument("-w", "--weight_decay", help="weight decay", type=float, default=1e-6)
parser.add_argument("-st", "--step_change_lr", help="step change learning rate", type=int, default=10)
parser.add_argument("-bs", "--batch_size", help="batch size", type=int, default=8)
parser.add_argument("-ep", "--epochs", help="epochs", type=int, default=30)
parser.add_argument("-l", "--label_path", help="epochs", type=str, default="dataset/labels.txt")
parser.add_argument("-d", "--device", help="device", type=str, default="cuda")
parser.add_argument("-o", "--optimizer", help="optimizer", type=str, default="adam")
parser.add_argument("-p", "--path", help="path to checkpoint", type=str, default=None)
parser.add_argument("--emb_size", help="embedding size", type=int, default=20)
parser.add_argument("--loss", help="loss type", type=str, default="hard_batch")

args = parser.parse_args()

CONFIG = get_config(args.config)
CONFIG.update(dict(vars(args))) 

label2int, int2label = get_labels(args.label_path)
num_classes = len(int2label)

def valid_epoch(X_train, y_train, X_test, y_test):
    global num_classes
    classifier = KNNClassifier()
    classifier.fit(X_train, y_train)
    train_acc = classifier.evaluate(X_train, y_train)
    valid_acc = classifier.evaluate(X_test, y_test)
    return train_acc, valid_acc, classifier

def train_one_epoch_hard_batch(epoch, train_loader, valid_loader, model, optimizer, scheduler, criterion, best_acc):
    device = args.device

    train_embs = []
    train_labels = []
    train_loss = []

    val_embs = []
    val_labels = []
    val_loss = []

    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (images, labels) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            images = images.to("cuda")
            embs = model(images)  
            loss = criterion(labels, embs.cpu())
            loss.to(device)
            loss.backward()
            optimizer.step()

            train_embs.extend(list(embs.cpu().detach().numpy()))
            train_labels.extend(list(labels.cpu().detach().numpy()))
            train_loss.append(loss.cpu().detach().numpy())
            tepoch.set_postfix(running_loss=np.mean(train_loss))
            del embs 
        scheduler.step()
    
    model.eval()
    for i, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        embs = model(images)  
        loss = criterion(labels, embs.cpu())
        val_embs.extend(list(embs.cpu().detach().numpy()))
        val_labels.extend(list(labels.cpu().detach().numpy()))
        val_loss.append(loss.cpu().detach().numpy())
        del embs

    train_acc, val_acc, classifier = valid_epoch(np.array(train_embs), np.array(train_labels), np.array(val_embs), np.array(val_labels))
    print("[DONE] Epoch {}/{} : train_loss: {:.4f}, val_loss: {:.4f}, train_acc: {:.4f}, val_acc: {:.4f}".format(epoch, args.epochs, np.mean(train_loss), np.mean(val_loss), train_acc, val_acc))
    
    if val_acc > best_acc:
        best_acc = val_acc
        model_path = f'{CONFIG["model_path"]}/extractor.pth'
        print("[INFO] Saving model {}...".format(model_path))
        torch.save({"model_state_dict": model.state_dict(),
            "optimzier_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        }, model_path)
        classifier.save_model(os.path.join(CONFIG["model_path"], "classifier.pkl"))

    return np.mean(train_loss), np.mean(val_loss), train_acc, val_acc, best_acc

def train_one_epoch(epoch, dataloader, model, optimizer, scheduler, criterion, loss_history):
    device = args.device
    with tqdm(dataloader, unit="batch") as tepoch:
        best_loss = 99999
        running_loss = []
        for i, (images, labels, _) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            anchors = torch.stack([imgs[0] for imgs in images], dim=0)
            positives = torch.stack([imgs[1] for imgs in images], dim=0)
            negatives = torch.stack([imgs[2] for imgs in images], dim=0)

            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            
            optimizer.zero_grad()
            anchor_emb = model(anchors)    
            pos_emb = model(positives)
            neg_emb = model(negatives)
            
            loss = criterion(anchor_emb, pos_emb, neg_emb)

            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
            tepoch.set_postfix(running_loss=np.mean(running_loss))

            del anchor_emb
            del pos_emb
            del neg_emb

        scheduler.step()

        print(f'Epoch {epoch}/{args.epochs} : Loss: {np.mean(running_loss)}')

        loss_history.append(np.mean(running_loss))
        
        if np.mean(running_loss) < best_loss:
            best_loss = np.mean(running_loss)
            model_path = f'{CONFIG["model_path"]}/trained_model_{epoch}.pth'
            print("[INFO] Saving model {}...".format(model_path))
            torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": {"loss": loss_history}
            }, model_path)
        plot_training_history(loss_history)
        return loss_history

def plot_history(train_loss, val_loss, train_acc, val_acc):
    fig1 = plt.figure(1)
    plt.title("Model Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_loss, color='blue')
    plt.plot(val_loss, color='orange')
    plt.legend(['train', 'val'], loc='upper right')
    fig1.savefig("{}/loss_history.png".format(CONFIG["model_path"]), dpi = 100, facecolor='white')

    fig2 = plt.figure(2)
    plt.title("Model Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.plot(train_acc, color='blue')
    plt.plot(val_acc, color='orange')
    plt.legend(['train', 'val'], loc='upper right')
    fig2.savefig("{}/acc_history.png".format(CONFIG["model_path"]), dpi = 100, facecolor='white')

    with open(f'{CONFIG["model_path"]}/logs.txt', 'a') as f:
        f.write("Epoch {}/{} : train_loss: {}, val_loss: {}, train_acc: {}, val_acc: {}\n".format(len(train_loss), args.epochs, train_loss[-1], val_loss[-1], train_acc[-1], val_acc[-1]))


def train():
    CONFIG['model_path'] = f'logs/{datetime.now().strftime("%m%d%Y_%H%M%S")}'
    if not os.path.exists(CONFIG["model_path"]):
        os.mkdir(CONFIG["model_path"])

    copyfile(args.label_path, f'{CONFIG["model_path"]}/labels.txt')
    save_config(CONFIG, f'{CONFIG["model_path"]}/config.yaml')

    model = Network(args.emb_size)
    model.to(args.device)
    model.train()

    if args.loss == "hard_batch":
        triplet_loss = HardBatchTripletLoss(0.8)
    else:
        triplet_loss = TripletLoss()

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_change_lr, gamma=0.5)

    trainset = TemplateDataset(CONFIG["train_data"], label2int, transforms=get_train_transforms())  
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    validset = TemplateDataset(CONFIG["valid_data"], label2int, transforms=get_train_transforms())  
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if (args.path):
        model, optimizer, scheduler, loss_history = load_checkpoint(checkpoint, model, optimizer, scheduler)
    history = {}
    history["train_loss"] = []
    history["val_loss"] = []
    history["train_acc"] = []
    history["val_acc"] = []
    best_acc = -1
    for epoch in range(args.epochs):
        train_loss, val_loss, train_acc, val_acc, best_acc = train_one_epoch_hard_batch(epoch, trainloader, validloader, model, optimizer, scheduler, triplet_loss, best_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        plot_history(**history)

if __name__ == "__main__":
    train()