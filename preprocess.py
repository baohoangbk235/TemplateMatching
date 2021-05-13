import cv2
from craft_text_detector import Craft
import numpy as np 
import warnings
import matplotlib.pyplot as plt
import os
import glob
import random
import yaml 
from google.colab.patches import cv2_imshow
from utils import *
from tqdm import tqdm
from PIL import Image
from imgaug import augmenters as iaa
import imgaug as ia
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--augment", help="data augmentation", type=bool, default=False)

args = parser.parse_args()

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
with open('config.yaml') as file:
    CONFIG = yaml.load(file, Loader=loader) 

def pipeline(ori_img, craft_net, refine_net, show_img=False):
    clt = KMeans(n_clusters = 3)
    removed_line, extracted_line = extract_line(ori_img)
    extracted_text = extract_text(removed_line, craft_net, refine_net, clt)
    final = extract_dotline(extracted_text, extracted_line)
    if show_img:
        visualize(["Origin", "Final"], [ori_img, final], 1, 2)
    return final

if __name__ == "__main__":
    if args.augment:
        aug = [iaa.AdditiveGaussianNoise(scale=0.1*255), iaa.Affine(scale=(0.7, 1.0), translate_percent=(-0.1, 0.1), order=[0, 1], cval=(0, 255), mode=ia.ALL, rotate=(-3, 3)),iaa.EdgeDetect(alpha=(0.0, 0.5)),
       iaa.PerspectiveTransform(scale=(0.01, 0.1)), ]
    
    refine_net = load_refinenet_model(cuda=True, weight_path=CONFIG["REFINER_WEIGHT"])
    craft_net = load_craftnet_model(cuda=True, weight_path=CONFIG["CRAFT_WEIGHT"])
    print('[INFO] Loading weight done!')
    os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)
    for class_ in os.listdir(CONFIG["DATA_DIR"]):
        print("[INFO] Processing class {}...".format(class_))        
        os.makedirs(os.path.join(CONFIG["OUT_DIR"], class_), exist_ok=True)
        class_path = os.path.join(CONFIG["DATA_DIR"], class_)
        img_filenames = random.choices(os.listdir(class_path), k=20)
        for img_filename in img_filenames:
            img_path = os.path.join(class_path, img_filename)
            ori_img = cv2.imread(img_path)
            final = pipeline(ori_img, craft_net, refine_net)
            cv2.imwrite(os.path.join(CONFIG["OUT_DIR"], class_, os.path.basename(img_path)), final)
            if args.augment:
                for i in range(10):
                    num_of_augment = random.randint(1, 3)
                    augments = random.sample(range(len(aug)), k=num_of_augment)
                    for j in augments:
                        result = aug[j].augment_image(final)
                    cv2.imwrite(os.path.join(CONFIG['OUT_DIR'], class_, img_filename[:-4] +  f'_{i}.jpg'), final)
    print("Preprocess done!")