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
from network import Network, Identity, TripletLoss
from dataset import TemplateDataset, collate_fn, get_train_transforms
from utils import get_labels, get_config

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="data augmentation", type=str, default="config.yaml")

args = parser.parse_args()

warnings.filterwarnings('ignore')

CONFIG = get_config(args.config)

def train_one_epoch(epoch, dataloader, model, optimizer, scheduler, criterion, loss_history):
    device = CONFIG["device"]
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

        print(f'Epoch {epoch}/{CONFIG["num_epochs"]} : Loss: {np.mean(running_loss)}')

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


def train():
    CONFIG['model_path'] = f'weights/{datetime.now().strftime("%m%d%Y_%H%M%S")}'
    if not os.path.exists(CONFIG["model_path"]):
        os.mkdir(CONFIG["model_path"])

    num_classes = len(os.listdir(CONFIG["train_data"]))
    label2int, int2label = get_labels(CONFIG['labels'])
    model = Network(20)
    model.to(CONFIG["device"])
    model.train()

    triplet_loss = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    scheduler = lr_scheduler.StepLR(optimizer, step_size=CONFIG["step_change_lr"], gamma=0.5)

    traindataset = TemplateDataset(CONFIG["train_data"], transforms=get_train_transforms(), show=False)  

    trainloader = DataLoader(traindataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)

    loss_history = []

    for epoch in range(CONFIG["num_epochs"]):
        train_one_epoch(epoch, trainloader, model, optimizer, scheduler, triplet_loss, loss_history)

    plot_training_history(loss_history)

if __name__ == "__main__":
    train()