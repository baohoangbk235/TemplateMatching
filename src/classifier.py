from network import Classifier, Network, get_trained_embedding
from dataset import EmbeddingDataset, TemplateDataset, collate_fn, collate_test, get_train_transforms
from utils import load_checkpoint, get_labels, get_config
import argparse
from torch.optim import lr_scheduler
import torch.optim as optim
import torch 
import torch.nn as nn 
from torch.utils.data import  DataLoader
from tqdm import tqdm 
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--model_path", help="path to checkpoint of feature extraction model", type=str)
parser.add_argument("-c", "--config", help="data augmentation", type=str, default="config.yaml")

args = parser.parse_args()

CONFIG = get_config(args.config)
CONFIG['model_path'] = args.model_path

def train_classifier():
    list_embs, list_labels, list_paths = get_trained_embedding(CONFIG)
    print("[INFO] Start training classifier...!")
    classy_dataset = EmbeddingDataset(list_embs, list_labels)
    loader = DataLoader(classy_dataset, batch_size=16, shuffle=True, collate_fn=collate_test)
    device = CONFIG["device"]
    net = Classifier()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-6)
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
    running_loss = []
    loss_history = []
    for epoch in range(100):
        with tqdm(loader, unit="batch") as tepoch:
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
                tepoch.set_postfix(running_loss=np.mean(running_loss))

            scheduler.step()
            loss_history.append(np.mean(running_loss))
            if epoch % 99 == 0:
                print('Epoch {} loss: {}'.format(epoch + 1, np.mean(running_loss)))
                torch.save({"model_state_dict": net.state_dict(),
                            "optimzier_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "history": {"loss": loss_history}
                        }, f'weights/classifier/classifier_epoch_{epoch}.pth')
            running_loss = []
    print('Finished Training')

if __name__ ==  "__main__":
    train_classifier(list_embs, list_labels)