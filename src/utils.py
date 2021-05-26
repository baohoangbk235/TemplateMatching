import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import os
import glob
import random
import warnings
import torch 
import yaml
import re 
from torch.optim import lr_scheduler
import pickle 
warnings.filterwarnings('ignore')

def load_checkpoint(checkpoint, model, optimizer, scheduler):
    config = torch.load(checkpoint)
    model.load_state_dict(config["model_state_dict"])
    optimizer.load_state_dict(config["optimzier_state_dict"])
    scheduler.load_state_dict(config["scheduler_state_dict"])
    return model, optimizer, scheduler, config['history']['loss']

def get_labels(labels_path):
    int2label = []
    with open(labels_path,'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            label = line.rstrip("\n").strip()
            if label != "":
                int2label.append(label)
    label2int = {k:v for v,k in enumerate(int2label)}
    return label2int, int2label

def get_config(config_path):
    with open(config_path, 'r') as f:
        print("[INFO] Loading {}...".format(config_path))
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        CONFIG = yaml.load(f, Loader=loader) 
        print(CONFIG)
    return CONFIG

def save_config(CONFIG, config_path):
    with open(config_path, 'w') as yaml_file:
        yaml.safe_dump(CONFIG, yaml_file, default_flow_style=False)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss