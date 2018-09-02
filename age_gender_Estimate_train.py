import chainer
import chainercv
import caffe
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import cupy as xp
import glob
from itertools import chain
import random
import re
from chainer.datasets import LabeledImageDataset
from chainer import datasets
from tqdm import tqdm_notebook
from chainer.datasets import TransformDataset
from itertools import chain

width = 128
height = 128

AGE_LABLES = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', '41-50', '51-60', '61-']#0,1,2,3,4,5,6,7,8,9
GENDER_LABLES = ['MALE', 'FEMALE']#0,1
#LABLES = [('1-5', 'MALE'), ('1-5', 'FEMALE')...('61-', 'MALE', ('61-', 'FEMALE'))]#0,1,2,..18,19

def transform(inputs):
    img, label = inputs
    img = img[:3, ...]
    img = img - mean[:, None, None]
    img = img.astype(np.float32)
    if np.random.rand() > 0.5:
        img = img[..., ::-1]
    return img, label

def load_image():
    num = 0
    print(num)
    filepaths = glob.glob('Dataset/Data/*.jpg')

    datasets = []
    for FP in filepaths:
        num = num + 1
        #if num == 100:
        #    break
        img = np.array(Image.open(FP).resize((128, 128)).convert('L'))
        try:
            label1 = int(re.split(r'_', re.split(r'/', FP)[2])[0])
            label2 = int(re.split(r'_', re.split(r'/', FP)[2])[1])
        except Exception:
            continue
        x = np.array(img, dtype=np.float32)
        #x = x.resize((3, 128, 128))
        t1 = np.array(label2, dtype=np.int32) # gender classificate
        t2 = np.array(label1, dtype=np.int32) # age classificate
        if 1 <= t2 <= 5:
            t = t1 + (0 * 2)
        elif 6 <= t2 <= 10:
            t = t1 + (1 * 2)
        elif 11 <= t2 <= 15:
            t = t1 + (2 * 2)
        elif 16 <= t2 <= 20:
            t = t1 + (3 * 2)
        elif 21 <= t2 <= 25:
            t = t1 + (4 * 2)
        elif 26 <= t2 <= 30:
            t = t1 + (5 * 2)
        elif 31 <= t2 <= 40:
            t = t1 + (6 * 2)
        elif 41 <= t2 <= 50:
            t = t1 + (7 * 2)
        elif 51 <= t2 <= 60:
            t = t1 + (8 * 2)
        elif 61 <= t2:
            t = t1 + (9 * 2)

        datasets.append((x,t))
        print(num)

    random.shuffle(datasets)
    return datasets

def train():
    global mean
    DS = load_image()
    d = LabeledImageDataset(DS)
    if not os.path.exists('Dataset/image_mean.npy'):
        t, _ = datasets.split_dataset_random(d, int(len(d) * 0.8), seed=0)
        mean = np.zeros((3, height, width))
        for img in t._dataset._pairs:
            i = img[0]
            mean += i
        mean = mean / float(len(d))
        np.save('Dataset/image_mean.npy', mean)
    else:
        mean = np.load('Dataset/image_mean.npy')
    td = TransformDataset(d, transform)
    return td, d, datasets

if __name__ == '__main__':
    td, d, DS = train()