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
from sklearn.cluster import KMeans
from craft_text_detector import (
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
warnings.filterwarnings('ignore')

def extract_line(src):
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ')
        return -1
    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    # [horiz]
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 10
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    horizontal = cv2.bitwise_not(horizontal)
    # show_wait_destroy("horizontal_bit", horizontal)
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 10
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    # Inverse vertical image
    vertical = cv2.bitwise_not(vertical)
    # show_wait_destroy("vertical_bit", vertical)
    '''
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img ///delete
    5. smooth.copyTo(src, edges)
    '''
    # vertical
    # Step 1
    vertical_edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 3, -2)
    # show_wait_destroy("edges", edges)
    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    vertical_edges = cv2.dilate(vertical_edges, kernel)
    # show_wait_destroy("dilate", edges)
    # Step 3
    vertical_smooth = np.copy(vertical)
    # Step 4
    # vertical_smooth = cv.blur(vertical_smooth, (2, 2))
    # Step 5
    (rows, cols) = np.where(vertical_edges != 0)
    vertical[rows, cols] = vertical_smooth[rows, cols]
    # Show final result
    # horizontal
    horizontal_edges = cv2.adaptiveThreshold(horizontal,255,cv2.ADAPTIVE_THRESH_MEAN_C, \
                                  cv2.THRESH_BINARY,3, -2)
    # show_wait_destroy("edges", edges1)
    kernel = np.ones((2, 2), np.uint8)
    horizontal_edges = cv2.dilate(horizontal_edges, kernel)
    # show_wait_destroy("dilate", edges1)
    # Step 3
    horizontal_smooth = np.copy(horizontal)
    # Step 4
    # horizontal_smooth = cv2.blur(horizontal_smooth, (2, 2))
    # Step 5
    (rows, cols) = np.where(horizontal_edges != 0)
    horizontal[rows, cols] = horizontal_smooth[rows, cols]
    # Show final result
    # show_wait_destroy("smooth - final", horizontal)
    # [smooth]
    # dot line
    # result
    extracted_line = cv2.bitwise_and(horizontal, vertical)
    line = 255 - extracted_line

    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(line ,kernel, iterations = 1)
    mask = np.reshape(np.repeat([dilation],3),(dilation.shape[0],dilation.shape[1],3))
    removed_line = cv2.bitwise_or(src,mask)

    return removed_line, extracted_line

def extract_dotline(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = bilateral_filter(gray)

    th = thresholding(gray, "gauss")
    th = (255-th)

    vertical_kernel = np.ones((15,1),np.uint8)
    horizontal_kernel = np.ones((1,40),np.uint8)
    dilation = cv2.dilate(th, horizontal_kernel,iterations = 1)

    # vertical_erosion = cv2.erode(th, vertical_kernel, iterations = 1)
    # horizontal_erosion = cv2.erode(th, horizontal_kernel, iterations = 1)

    # erosion = cv2.bitwise_or(vertical_erosion, horizontal_erosion, mask = None)   
    contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    copy_img = img.copy()

    filter_contours = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if h < 10 or w < 10:
            filter_contours.append(cnt)
            # cv2.rectangle(copy_img, (y,y+h), (x,x+w), (0,255,0), 2)

    mask = np.ones(img.shape[:2], dtype="uint8") * 0
    cv2.drawContours(mask, filter_contours, -1, 255, -1)   
    final = cv2.bitwise_and(img, img, mask=mask)
    # titles = ['Original Image', 'Erosion', 'Contours', 'Final']
    # visualize(titles, [img, gray, th, final])
    
    return final, mask

def filter(coords, shape):
    a = []
    for i, c in enumerate(coords):
        if c < 0:
            c = 0
        if (i == 2) or (i == 0):
            if c > shape[1] :
                c = shape[1]
        if (i == 1) or (i == 3):
            if c > shape[0] :
                c = shape[0]
        a.append(int(c))
    return a

def extract_text(img, craft_net, refine_net, clt):
    prediction_result = get_prediction(
                image=img,
                craft_net=craft_net,
                # refine_net=refine_net,
                text_threshold=0.01,
                link_threshold=0.4,
                low_text=0.4,
                cuda=True,
                long_size=1280
            )
    
    color = get_bg_color(img, clt)

    for box in prediction_result['boxes']:
        x1, y1 = np.min(box, axis=0)
        x2, y2 = np.max(box, axis=0)
        h,w = y2-y1, x2-x1
        
        roi = [x1, y1-100, x2, y2+100]
        roi = filter(roi, img.shape)
        a = filter([x1,y1,x2,y2], img.shape)

        color = get_bg_color(img[roi[1]:roi[3],:], clt)

        if (a[2]-a[0]) * (a[3]-a[1]) <  img.shape[0] * img.shape[1] * 0.03:
            img[a[1]-1: a[3]+2, a[0]-1: a[2]+2] = color
    return img

def extract_dotline(extracted_text, extracted_line):
    gray = cv2.cvtColor(extracted_text, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    #xử lý thêm, extract dotline chưa tốt
    extracted_dotline = cv2.bitwise_and(th, extracted_line)
    img = cv2.bitwise_and(extracted_text, extracted_text, mask=extracted_dotline)
    return extracted_dotline

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist

def get_bg_color(image, clt):
    bgr_planes = cv2.split(image)
    histSize = 256
    histRange = (0, 256) # the upper boundary is exclusive
    bg_color = []
    for i in range(3):
        hist = cv2.calcHist(bgr_planes, [i], None, [histSize], histRange)
        bg_color.append(np.argmax(hist))
    return tuple(bg_color)

def visualize(titles, images, row=2, col=2):
    assert len(titles) == len(images), "Titles length and images length are not equal"
    assert len(images) == (row * col), "Num of sub fig and length of images are not equal"
    plt.figure(figsize=(60,40))
    for i in range(row * col):
        plt.subplot(row, col, i+1)
        if len(images[i].shape) == 2:
            plt.imshow(images[i].astype("uint8"), "gray")
        else:   
            plt.imshow(cv2.cvtColor(images[i].astype("uint8"), cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
    plt.show()

def pipeline(ori_img, craft_net, refine_net, show_img=False):
    clt = KMeans(n_clusters = 3)
    removed_line, extracted_line = extract_line(ori_img)
    extracted_text = extract_text(removed_line, craft_net, refine_net, clt)
    final = extract_dotline(extracted_text, extracted_line)
    if show_img:
        visualize(["Origin", "Final"], [ori_img, final], 1, 2)
    return final