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
initial_lr = 0.01
lr_drop_epoch = 10
lr_drop_ratio = 0.1
train_epoch = 20

AGE_LABLES = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', '41-50', '51-60', '61-']#0,1,2,3,4,5,6,7,8,9
GENDER_LABLES = ['MALE', 'FEMALE']#0,1
#LABLES = [('1-5', 'MALE'), ('1-5', 'FEMALE')...('61-', 'MALE', ('61-', 'FEMALE'))]#0,1,2,..18,19

class Illust2Vec(Chain):

    CAFFEMODEL_FN = 'illust2vec_ver200.caffemodel'

    def __init__(self, n_classes, unchain=True):
        w = chainer.initializers.HeNormal()
        model = CaffeFunction(self.CAFFEMODEL_FN)  # CaffeModelを読み込んで保存します。（時間がかかります）
        del model.encode1  # メモリ節約のため不要なレイヤを削除します。
        del model.encode2
        del model.forwards['encode1']
        del model.forwards['encode2']
        model.layers = model.layers[:-2]

        super(Illust2Vec, self).__init__()
        with self.init_scope():
            self.trunk = model  # 元のIllust2Vecモデルをtrunkとしてこのモデルに含めます。
            self.fc7 = L.Linear(None, 4096, initialW=w)
            self.bn7 = L.BatchNormalization(4096)
            self.fc8 = L.Linear(4096, n_classes, initialW=w)

    def __call__(self, x):
        h = self.trunk({'data': x}, ['conv6_3'])[0]  # 元のIllust2Vecモデルのconv6_3の出力を取り出します。
        h.unchain_backward()
        h = F.dropout(F.relu(self.bn7(self.fc7(h))))  # ここ以降は新しく追加した層です。
        return self.fc8(h)

def resize(img):
    img = Image.fromarray(img.transpose(1, 2, 0))
    img = img.resize((width, height), Image.BICUBIC)
    return np.asarray(img).transpose(2, 0, 1)

# 各データに行う変換
def transform(inputs):
    img, label = inputs
    img = img[:3, ...]
    img = resize(img.astype(np.uint8))
    img = img - mean[:, None, None]
    img = img.astype(np.float32)
    # ランダムに左右反転
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
        img = FP
        try:
            label1 = int(re.split(r'_', re.split(r'/', FP)[2])[0])
            label2 = int(re.split(r'_', re.split(r'/', FP)[2])[1])
        except Exception:
            continue
        #if num == 1000:
        #    break
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
    train, valid = datasets.split_dataset_random(td, int(len(d) * 0.8), seed=0)
    mean = mean.mean(axis=(1, 2))

    n_classes = 20
    model = Illust2Vec(n_classes)
    model = L.Classifier(model)

    train_iter = iterators.MultiprocessIterator(train, batchsize)
    valid_iter = iterators.MultiprocessIterator(valid, batchsize, repeat=False, shuffle=False)

    optimizer = optimizers.MomentumSGD(lr=initial_lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    trainer = training.Trainer(updater, (train_epoch, 'epoch'), out='AnimeFace-result')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    # 標準出力に書き出したい値
    trainer.extend(extensions.PrintReport(
        ['epoch',
         'main/loss',
         'main/accuracy',
         'val/main/loss',
         'val/main/accuracy',
         'elapsed_time',
         'lr']))

    # ロスのプロットを毎エポック自動的に保存
    trainer.extend(extensions.PlotReport(
        ['main/loss',
         'val/main/loss'],
        'epoch', file_name='loss.png'))

    # 精度のプロットも毎エポック自動的に保存
    trainer.extend(extensions.PlotReport(
        ['main/accuracy',
         'val/main/accuracy'],
        'epoch', file_name='accuracy.png'))

    # モデルのtrainプロパティをFalseに設定してvalidationするextension
    trainer.extend(extensions.Evaluator(valid_iter, model, device=gpu_id), name='val')

    # 指定したエポックごとに学習率をlr_drop_ratio倍にする
    trainer.extend(
        extensions.ExponentialShift('lr', lr_drop_ratio),
        trigger=(lr_drop_epoch, 'epoch'))

    trainer.run()
    return td, d, datasets

if __name__ == '__main__':
    td, d, DS = Train_main()