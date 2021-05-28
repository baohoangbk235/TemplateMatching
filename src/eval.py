from network import Classifier, Network
from dataset import EmbeddingDataset, TemplateDataset, collate_fn, get_train_transforms, get_test_transforms,collate_fn_test
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
from sklearn.metrics import accuracy_score
from collections import Counter
import cv2 
import matplotlib.pyplot as plt 
import os
import pickle 
import shutil
from lib import pipeline
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
from PIL import Image

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
    def __init__(self, model_dir, result_dir="results/"):
        self.model_dir = model_dir
        self.CONFIG = get_config(os.path.join(model_dir, "config.yaml"))
        self.embSize = self.CONFIG["emb_size"]
        self.device = self.CONFIG["device"]
        self.label2int, self.int2label = get_labels(os.path.join(model_dir, "labels.txt"))
        self.result_dir = result_dir
        self.createResultDir()
        self.featureExtractorPath = os.path.join(model_dir, "extractor.pth")
        self.featureExtractor = self.getModel()
        self.classifierPath = os.path.join(model_dir, "classifier.pkl")
        self.classifier = self.get_classifier()
        self.craft_net = load_craftnet_model(cuda=True, weight_path=self.CONFIG["CRAFT_WEIGHT"])
        self.threshold = [
            0.15658274643186937,
            0.36065899509540494,
            0.38856512542345073,
            0.40749227647561814,
            0.16654823949181954,
            0.1937299593585302,
            0.11403658569607081,
            0.45526403121798326,
        ]
    
    def get_classifier(self):
        classifier = KNNClassifier()
        classifier.load_model(self.classifierPath)
        return classifier

    def getModel(self):
        config = torch.load(self.featureExtractorPath)
        model = Network(self.embSize)
        model.load_state_dict(config["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model 
    
    def createResultDir(self):
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        clearFolder(f'{self.result_dir}/images/')
        clearFolder(f'{self.result_dir}/trues/')
        clearFolder(f'{self.result_dir}/falses/')

    def getEmbs(self, dataPath, valid=False):
        list_embs = []
        list_paths = []
        list_labels = []
        self.featureExtractor.to(self.device)
        if not valid:
            print("Testing ...")
            dataset = TemplateDataset(dataPath, self.label2int, transforms=get_test_transforms(), mode="test") 
            loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn_test)
            for i, (images, paths) in enumerate(loader):
                anchors = torch.stack(images, axis=0)
                anchors = anchors.to(args.device)
                anchor_emb = self.featureExtractor(anchors)  
                list_embs.extend(list(anchor_emb.cpu().detach().numpy()))
                list_paths.extend(paths)
                del anchor_emb
            self.featureExtractor.to("cpu")
            return list_embs, list_paths
        else:
            print("Evaluating ...")
            dataset = TemplateDataset(dataPath, self.label2int, transforms=get_test_transforms(), mode="valid") 
            loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
            for i, (images, labels,  paths) in enumerate(loader):
                anchors = torch.stack(images, axis=0)
                anchors = anchors.to(self.device)
                anchor_emb = self.featureExtractor(anchors)  
                list_embs.extend(list(anchor_emb.cpu().detach().numpy()))
                list_labels.extend(labels)
                list_paths.extend(paths)
                del anchor_emb
            self.featureExtractor.to("cpu")
            return list_embs, list_labels, list_paths

    def run(self, dataPath, valid=False):
        print("[INFO] Calculating embeddings ...")
        if valid:
            embs, labels, paths = self.getEmbs(dataPath, valid)
            with open(os.path.join(self.model_dir, 'X.pkl'), 'wb') as f:
                pickle.dump(embs, f)
            with open(os.path.join(self.model_dir, 'y.pkl'), 'wb') as f:
                pickle.dump(labels, f)
            with open(os.path.join(self.model_dir, 'path.pkl'), 'wb') as f:
                pickle.dump(paths, f)  
        else:
            embs, paths = self.getEmbs(dataPath, valid)

        print("[INFO] Classifying embeddings ...")

        predicts, probs = self.classifier.test(embs)

        print("[INFO] Visualizing results ...")
        if valid:
            acc = accuracy_score(predicts, labels)
            print(f'Valid Accuracy: {acc}')
            self.compareToVisualize(predicts, probs, labels, paths)
        
        else:
            for (pred, prob, path) in zip(predicts, probs, paths):
                pred_label = self.int2label[pred]
                filename = os.path.basename(path)
                ori_path = path.replace("preprocessed_", "")
                shutil.copy(ori_path, os.path.join(resultDir, pred_label, filename))

    def compareToVisualize(self, predicts, probs, groundTruths, paths):
        for i in range(len(predicts)):
            ori_path = paths[i].replace("preprocessed_", "")
            img = cv2.imread(ori_path)
            if img is None:
                continue
            plt.figure(figsize=(20,20))
            plt.xlabel("{} - Ground truth: {} - Predict: {}".format(ori_path, self.int2label[groundTruths[i]], self.int2label[predicts[i]]), fontsize=25)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            
            filename = os.path.basename(paths[i])
            if (predicts[i] == groundTruths[i]):
                plt.savefig(f'{self.result_dir}/trues/{filename}')
            else:
                plt.savefig(f'{self.result_dir}/falses/{filename}')

    def predict(self, img):
        preprocessed = pipeline(img, self.craft_net)
        self.featureExtractor.to(self.device)
        preprocessed = np.stack([preprocessed, preprocessed, preprocessed], axis=2)
        transformed = get_train_transforms()(image=preprocessed)["image"]
        emb = self.featureExtractor(transformed.unsqueeze(0).to(self.device))
        self.featureExtractor.to("cpu")
        predict, distance = self.classifier.test(emb.cpu().detach().numpy())
        if distance[0][0] > self.threshold[predict[0]]:
            label = "Unknown"
        else:
            label = self.int2label[predict[0]]
        print("result: {} score: {}\n{}, {}".format(label, distance[0][0], self.int2label[predict[0]], self.threshold[predict[0]]))
        return label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--model_dir", help="path to checkpoint of feature extraction model", type=str)
    parser.add_argument("-i", "--input_path", help="input", type=str)
    parser.add_argument("--valid", help="valid or test", type=bool, default=False)
    args = parser.parse_args()
    templateClassifier = TemplateClassifier(args.model_dir)
    if os.path.exists(args.input_path):
        if os.path.isdir(args.input_path):
            if args.valid:
                templateClassifier.run(args.input_path, valid=True)
            else:
                templateClassifier.run(args.input_path, valid=False)
        else:
            img = cv2.imread(args.input_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            templateClassifier.predict(img)
    else:
        raise Exception("The file doesnt exist")