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
import shutil
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
from lib import pipeline
from utils import get_config



def apply_augment(image, aug):
    results = []
    num_of_augment = random.randint(1, 3)
    augments = random.sample(range(len(aug)), k=num_of_augment)
    for j in augments:
        result = aug[j].augment_image(image)
        results.append(result)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="path to checkpoint of feature extraction model", type=str)
    parser.add_argument("--config", help="path to checkpoint of feature extraction model", type=str, default="config.yaml")
    parser.add_argument("--mode", help="mode train model", type=str, default="test")
    args = parser.parse_args()
    CONFIG = get_config(args.config)

    aug = [iaa.AdditiveGaussianNoise(scale=0.1*255), iaa.Affine(scale=(0.7, 1.0), translate_percent=(-0.1, 0.1), order=[0, 1], cval=(0, 255), mode=ia.ALL, rotate=(-3, 3)),iaa.EdgeDetect(alpha=(0.0, 0.5)),
                iaa.PerspectiveTransform(scale=(0.01, 0.1)), ]
    craft_net = load_craftnet_model(cuda=True, weight_path=CONFIG["CRAFT_WEIGHT"])
    print('[INFO] Loading weight done!')

    input_dir = args.input_dir 
    last_path = os.path.basename(os.path.normpath(input_dir))
    output_dir = input_dir.replace(last_path, "preprocessed_" + last_path)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    if args.mode == "test":
        for image_path in glob.glob(os.path.join(input_dir, '*')):
            ori_img = cv2.imread(image_path)
            final = pipeline(ori_img, craft_net)
            filename = os.path.basename(image_path)
            cv2.imwrite(os.path.join(output_dir, filename), final)
    else:
        for class_ in os.listdir(input_dir):
            print("[INFO] Processing class {}...".format(class_))        
            os.makedirs(os.path.join(output_dir, class_), exist_ok=True)
            class_path = os.path.join(input_dir, class_)
            img_filenames = random.choices(os.listdir(class_path), k=20)
            for img_filename in img_filenames:
                img_path = os.path.join(class_path, img_filename)
                ori_img = cv2.imread(img_path)
                final = pipeline(ori_img, craft_net)
                cv2.imwrite(os.path.join(output_dir, class_, os.path.basename(img_path)), final)
                if args.mode == "train":
                    for i in range(10):
                        results = apply_augment(final, aug)
                        for result in results:
                            cv2.imwrite(os.path.join(output_dir, class_, img_filename[:-4] +  f'_{i}.jpg'), result)
        

    print("Preprocess done!")