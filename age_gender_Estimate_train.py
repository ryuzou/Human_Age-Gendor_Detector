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
from chainer import Chain
from chainer import serializers

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import Chain
from chainer.links.caffe import CaffeFunction
from chainer import serializers

from chainer import iterators
from chainer import training
from chainer import optimizers
from chainer.training import extensions
from chainer.training import triggers
from chainer.dataset import concat_examples

width = 128
height = 128

batchsize = 64
gpu_id = 0
max_epoch = 10

AGE_LABLES = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', '41-50', '51-60', '61-']#0,1,2,3,4,5,6,7,8,9
GENDER_LABLES = ['MALE', 'FEMALE']#0,1
#LABLES = [('1-5', 'MALE'), ('1-5', 'FEMALE')...('61-', 'MALE', ('61-', 'FEMALE'))]#0,1,2,..18,19

counter = 0

class MLP(Chain):

    def __init__(self, n_mid_units=100, n_out=20):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def resize(img):
    global counter
    print(counter)
    counter = counter + 1
    img = Image.fromarray(img.transpose(1, 2, 0))
    img = img.resize((width, height), Image.BICUBIC)
    return np.asarray(img).transpose(2, 0, 1)

def transform(inputs):
    img, label = inputs
    img = img[:3, ...]
    img = resize(img.astype(np.uint8))
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
        if num >= 2000:
            break
        try:
            label1 = int(re.split(r'_', re.split(r'/', FP)[2])[0])
            label2 = int(re.split(r'_', re.split(r'/', FP)[2])[1])
        except Exception:
            continue
        try:
            resize(np.array(Image.open(FP))[:3, ...].astype(np.uint8))
        except Exception:
            continue
        img = FP
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

        datasets.append((img,t))
        print(num)

    random.shuffle(datasets)
    return datasets

def Train_main():
    global mean
    DS = load_image()
    d = LabeledImageDataset(DS)
    if not os.path.exists('Dataset/image_mean.npy'):
        print("caluculating mean")
        t, _ = datasets.split_dataset_random(d, int(len(d) * 0.8), seed=0)
        mean = np.zeros((3, height, width))
        for img in t:
            try:
                img = resize(img[0].astype(np.uint8))
                mean += img
            except Exception:
                continue
        mean = mean / float(len(d))
        np.save('Dataset/image_mean.npy', mean)
    else:
        mean = np.load('Dataset/image_mean.npy')
    td = TransformDataset(d, transform)
    train, test = datasets.split_dataset_random(td, int(len(d) * 0.8), seed=0)
    mean = mean.mean(axis=(1, 2))
    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, False, False)
    model = MLP()
    model = L.Classifier(model)
    model.to_gpu(gpu_id)
    optimizer = optimizers.MomentumSGD()
    optimizer.setup(model)
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')
    trainer.extend(extensions.LogReport())

    #Extentions
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()
    return td, d, datasets

if __name__ == '__main__':
    td, d, DS = Train_main()