from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from network import Classifier, Network
from dataset import EmbeddingDataset, TemplateDataset, collate_fn, get_train_transforms
from utils import load_checkpoint, get_labels, get_config
import argparse
from torch.optim import lr_scheduler
import torch.optim as optim
import torch 
import torch.nn as nn 
from torch.utils.data import  DataLoader
from tqdm import tqdm 
import numpy as np 
from utils import EarlyStopping
from abc import ABC 
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import pickle 

class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass 
    
    @abstractmethod
    def evaluate(self, X, y):
        pass 
    
    @abstractmethod
    def test(self, X_test):
        pass 

class KNNClassifier(BaseClassifier):
    def __init__(self, n_neighbors=1):
        super().__init__()
        self.nbrs = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X_train, y_train):
        self.nbrs.fit(X_train, y_train)

    def evaluate(self, X, y):
        return self.nbrs.score(X, y)

    def test(self, X_test):
        return self.nbrs.predict(X_test), self.nbrs.kneighbors(X_test)[0]
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.nbrs, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.nbrs = pickle.load(f)

class KMeansClassifier(BaseClassifier):
    def __init__(self, n_clusters):
        super().__init__()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    def fit(self, X_train, y_train):
        self.kmeans.fit(X_train, y_train)
        return self.kmeans.labels_

    def evaluate(self, X, y):
        return self.nbrs.score(X, y)

    def test(self, X_test):
        return self.nbrs.predict(X_test), self.nbrs.kneighbors(X_test)[0]
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.nbrs, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.nbrs = pickle.load(f)


class FCNClassifier(BaseClassifier):
    def __init__(self, input_dim, opt_type="adam", epochs=1000, device="cuda", patience=None, model_path=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.patience = patience
        self.opt_type = opt_type
        self.epochs = epochs
        self.model_path = model_path
        self.model = None 
        self.optimizer = None 
        self.scheduler = None 
        self.model, self.optimizer, self.scheduler = self.get_model()

    def get_model(self):
        net = Classifier(self.input_dim)
        net = net.to(self.device)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        if self.opt_type == "adam":
            optimizer = optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-6)
        else:
            optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
        return net, optimizer, scheduler

    def fit(self, X_train, y_train):
        print("[INFO] Start training classifier...!")
        traindataset = EmbeddingDataset(X_train, y_train)
        trainloader = DataLoader(traindataset, batch_size=16, shuffle=True, collate_fn=collate_test)
        criterion = nn.CrossEntropyLoss()
        running_loss = []
        loss_history = []
        with open("weights/classifier/logs.txt", "a+") as f:
            best_loss = 9999
            for epoch in range(self.epochs):
                self.model.train()
                for i, (X, yhat) in enumerate(trainloader):
                    X = torch.stack([x for x in X], dim=0)
                    
                    yhat = torch.tensor(yhat, dtype=torch.long)
                    yhat = torch.stack([y for y in yhat], dim=0)
                    
                    X = X.to(self.device)
                    yhat = yhat.to(self.device)

                    self.optimizer.zero_grad()
                    y = self.model(X)
                    loss = criterion(y, yhat)
                    loss.backward()
                    self.optimizer.step()
                    running_loss.append(loss.item())
                self.scheduler.step()
                loss_history.append(np.mean(running_loss))
            
                f.write('Epoch {} loss: {}\n'.format(epoch, np.mean(running_loss)))
            
                if np.mean(running_loss) < best_loss:
                    best_loss = np.mean(running_loss)
                    torch.save({"model_state_dict": self.model.state_dict(),
                                "optimzier_state_dict": self.optimizer.state_dict(),
                                "scheduler_state_dict": self.scheduler.state_dict()
                            }, f'weights/classifier/best.pth')
                self.model_path = 'weights/classifier/best.pth'
                if epoch % 100 == 0:
                    print('Epoch {} loss: {}, best_loss: {}'.format(epoch, np.mean(running_loss), best_loss))
                running_loss = []
        print('[INFO] Finished Training')

    def evaluate(self, X_val, y_val):
        if self.model_path is not None:
            config = torch.load(self.model_path)
            self.model.load_state_dict(config["model_state_dict"])
            self.optimizer.load_state_dict(config["optimzier_state_dict"])
            self.scheduler.load_state_dict(config["scheduler_state_dict"])
            
        dataset = EmbeddingDataset(X_val, y_val)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_test)
        self.model.eval()
        correct = 0
        total = 0
        for i, (X, yhat) in enumerate(loader):
            X = torch.stack([x for x in X], dim=0)
            
            yhat = torch.tensor(yhat, dtype=torch.long)
            yhat = torch.stack([y for y in yhat], dim=0)
            
            X = X.to(self.device)
            yhat = yhat.to(self.device)
            y = self.model(X)

            _, predicted = torch.max(y.data, 1)
            correct += (predicted  == yhat).sum().item()
            total += yhat.size(0)

        acc =  correct/total
        return acc 
    
    def test(self, X_test):
        if self.model_path is not None:
            config = torch.load(self.model_path)
            self.model.load_state_dict(config["model_state_dict"])
            self.optimizer.load_state_dict(config["optimzier_state_dict"])
            self.scheduler.load_state_dict(config["scheduler_state_dict"])
        
        self.model.eval()
        results = []
        for i, (X, yhat) in enumerate(loader):
            X = torch.stack([x for x in X], dim=0)
            
            yhat = torch.tensor(yhat, dtype=torch.long)
            yhat = torch.stack([y for y in yhat], dim=0)
            
            X = X.to(device)
            yhat = yhat.to(device)
            y = self.model(X)

            _, predicted = torch.max(y.data, 1)
            results.extend(list(predicted.cpu().detach().numpy()))
        
        return np.array(results)

