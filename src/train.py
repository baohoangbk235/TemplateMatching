import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import  DataLoader

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
from dataset import TemplateDataset, collate_fn, get_train_transforms, MyDataset
from utils import get_labels, get_config, save_config
from tensorboard_logger import log_value, configure
from shutil import copyfile



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
parser.add_argument("--loss", help="embedding size", type=str, default="hard_batch")

args = parser.parse_args()

warnings.filterwarnings('ignore')

CONFIG = get_config(args.config)
CONFIG["emb_size"] = args.emb_size
CONFIG["config"] = vars(args)

def train_one_epoch_hard_batch(epoch, dataloader, model, optimizer, scheduler, criterion, loss_history):
    device = args.device
    with tqdm(dataloader, unit="batch") as tepoch:
        best_loss = 99999
        running_loss = []
        for i, (images, labels) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            images = images.to("cuda")
            embs = model(images)  
            loss = criterion(labels, embs.cpu())
            loss.to(device)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
            tepoch.set_postfix(running_loss=np.mean(running_loss))
            del embs 

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

def plot_training_history(loss_history):
    fig = plt.figure()
    plt.title("Training")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss_history)
    fig.savefig("{}/history.png".format(CONFIG["model_path"]), dpi = 100, facecolor='white')
    with open(f'{CONFIG["model_path"]}/logs.txt', 'a') as f:
        f.write(f'Epoch {len(loss_history)}/{args.epochs} : Loss: {loss_history[-1]}\n')


def train():
    CONFIG['model_path'] = f'weights/{datetime.now().strftime("%m%d%Y_%H%M%S")}'
    if not os.path.exists(CONFIG["model_path"]):
        os.mkdir(CONFIG["model_path"])

    label2int, int2label = get_labels(args.label_path)
    copyfile(args.label_path, f'{CONFIG["model_path"]}/labels.txt')
    save_config(CONFIG, f'{CONFIG["model_path"]}/config.yaml')
    num_classes = len(int2label)

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

    # traindataset = TemplateDataset(CONFIG["train_data"], label2int, transforms=get_train_transforms(), show=False)  
    traindataset = MyDataset(CONFIG["train_data"], label2int, transforms=get_train_transforms())  
    trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    loss_history = []

    if (args.path):
            model, optimizer, scheduler, loss_history = load_checkpoint(checkpoint, model, optimizer, scheduler)

    for epoch in range(args.epochs):
        # train_one_epoch(epoch, trainloader, model, optimizer, scheduler, triplet_loss, loss_history)
        train_one_epoch_hard_batch(epoch, trainloader, model, optimizer, scheduler, triplet_loss, loss_history)

    plot_training_history(loss_history)

if __name__ == "__main__":
    train()