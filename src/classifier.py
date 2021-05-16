from network import Classifier, Network, save_trained_embedding
from dataset import EmbeddingDataset, TemplateDataset, collate_fn, collate_test, get_train_transforms
from utils import load_checkpoint, get_labels, get_config, get_trained_embeddings
import argparse
from torch.optim import lr_scheduler
import torch.optim as optim
import torch 
import torch.nn as nn 
from torch.utils.data import  DataLoader
from tqdm import tqdm 
import numpy as np 
from utils import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--model_path", help="path to checkpoint of feature extraction model", type=str)
parser.add_argument("-c", "--config", help="data augmentation", type=str, default="config.yaml")
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=5e-4)
parser.add_argument("-w", "--weight_decay", help="weight decay", type=float, default=1e-6)
parser.add_argument("-st", "--step_change_lr", help="step change learning rate", type=int, default=10)
parser.add_argument("-bs", "--batch_size", help="batch size", type=int, default=8)
parser.add_argument("-ep", "--epochs", help="epochs", type=int, default=1000)
parser.add_argument("-l", "--label_path", help="epochs", type=str, default="dataset/labels.txt")
parser.add_argument("-d", "--device", help="device", type=str, default="cuda")
parser.add_argument("-o", "--optimizer", help="optimizer", type=str, default="sgd")
args = parser.parse_args()

CONFIG = get_config(args.config)


def train_classifier():
    data = get_trained_embeddings()
    print("[INFO] Start training classifier...!")
    classy_dataset = EmbeddingDataset(data["train"][0], data["train"][1])
    loader = DataLoader(classy_dataset, batch_size=16, shuffle=True, collate_fn=collate_test)
    device = args.device
    net = Classifier(10)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=10, verbose=True)
    if args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-6)
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
    running_loss = []
    loss_history = []
    for epoch in range(args.epochs):
        correct = 0
        total = 0
        for i, (X, yhat) in enumerate(loader):
            X = torch.stack([x for x in X], dim=0)
            
            yhat = torch.tensor(yhat, dtype=torch.long)
            yhat = torch.stack([y for y in yhat], dim=0)
            
            X = X.to(device)
            yhat = yhat.to(device)

            optimizer.zero_grad()
            y = net(X)
            loss = criterion(y, yhat)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            _, predicted = torch.max(y.data, 1)
            correct += (predicted  == yhat).sum().item()
            total += yhat.size(0)
        
        # scheduler.step()
        loss_history.append(np.mean(running_loss))
        if epoch % 100 == 0:
            print('Epoch {} loss: {}, acc: {}'.format(epoch, np.mean(running_loss), correct/total))
            torch.save({"model_state_dict": net.state_dict(),
                        "optimzier_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "history": {"loss": loss_history}
                    }, f'weights/classifier/classifier_epoch_{epoch}.pth')
        running_loss = []
        # early_stopping(valid_loss, model)
        
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
    print('Finished Training')

if __name__ ==  "__main__":
    train_classifier()