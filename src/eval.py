from network import Classifier, Network
from dataset import EmbeddingDataset, TemplateDataset, collate_fn, get_train_transforms, collate_fn_test
from utils import load_checkpoint, get_labels, get_config
from Models.Classifier import KNNClassifier, FCNClassifier
import yaml 
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
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--model_path", help="path to checkpoint of feature extraction model", type=str)
parser.add_argument("-c", "--config", help="data augmentation", type=str, default="config.yaml")
parser.add_argument("-l", "--label_path", help="epochs", type=str, default="dataset/labels.txt")
parser.add_argument("-d", "--device", help="device", type=str, default="cuda")
parser.add_argument("--classifier", help="classifier path", type=str)
parser.add_argument("--valid", help="valid or test", type=bool, default=False)
parser.add_argument("--emb_size", help="embedding size", type=int, default=9)
args = parser.parse_args()

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def clearFolder(pathToFold):
    if not os.path.exists(pathToFold):
        os.mkdir(pathToFold)
    else:
        for filename in os.listdir(pathToFold):
            file_path = os.path.join(pathToFold, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

class TemplateClassifier():
    def __init__(self, model_dir):
        self.model_dir = model_dir

        self.CONFIG = get_config(os.path.join(model_dir, "config.yaml"))
        self.embSize = self.CONFIG["emb_size"]
        self.device = self.CONFIG["device"]
        self.label2int, self.int2label = get_labels(os.path.join(model_dir, "labels.txt"))
        
        self.featureExtractorPath = os.path.join(model_dir, "extractor.pth")
        self.featureExtractor = self.getModel()

        self.classifierPath = os.path.join(model_dir, "classifier.pkl")
        self.classifier = pickle.load(open(self.classifierPath, 'rb'))
    
    def getModel(self):
        config = torch.load(self.featureExtractorPath)
        model = Network(self.embSize)
        model.load_state_dict(config["model_state_dict"])
        model.to(self.device)
        return model 

    def getEmbs(self, dataPath):
        self.featureExtractor.eval()
        dataset = TemplateDataset(dataPath, self.label2int, transforms=get_train_transforms(), mode="test") 
        loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_test)

        list_embs = []
        list_paths = []

        for i, (images, paths) in enumerate(loader):
            anchors = torch.stack(images, axis=0)
            anchors = anchors.to(args.device)
            anchor_emb = self.featureExtractor(anchors)  
            list_embs.extend(list(anchor_emb.cpu().detach().numpy()))
            list_paths.extend(paths)
            del anchor_emb
        
        return list_embs, list_paths
        # data = {
        #     "X": np.array(list_embs),
        #     "y": np.array(list_labels),
        #     "path": list_paths
        # }
        # with open(outPath, "wb") as f:
        #     pickle.dump(data, f)
    
    # def getEmbeddings(self, embPath):
    #     with open(embPath, 'rb') as f:
    #         data = pickle.load(f)
    #     embs, labels, paths = data["X"], data["y"], data["path"]
    #     return embs, labels, paths
    def run(self, testPath):
        resultDir = 'results/images/'
        for label in self.label2int.keys():
            clearFolder(os.path.join(resultDir, label))

        valid = False 
        if valid:
            embs, labels, paths = self.getEmbs(testPath)
        else:
            embs, paths = self.getEmbs(testPath)
        
        predicts = self.classifier.test(embs)
        for (pred, path) in zip(predicts, paths):
            pred_label = self.int2label[pred]
            filename = os.path.basename(path)
            ori_path = path.replace("preprocessed_", "")
            shutil.copy(ori_path, os.path.join(resultDir, pred_label, filename))

        if valid:
            clearFolder("results/trues/")
            clearFolder("results/falses/")
            acc = classifier.evaluate(embs, labels)
            compareToVisualize(predicts, labels, paths)
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



if __name__ == "__main__":
    templateClassifier = TemplateClassifier(args.model_path)
    templateClassifier.run("/content/drive/MyDrive/DATN/dataset/preprocessed_test")