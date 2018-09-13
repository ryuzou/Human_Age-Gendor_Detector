import chainer
import chainer.links as L
from chainer import Chain
import chainer.functions as F
import numpy as np
from PIL import Image
from chainer.links import VGG16Layers
import cupy
import glob
import tqdm

caffemodel = './VGG_ILSVRC_16_layers.caffemodel'
chainermodel = './VGG_ILSVRC_16_layers.npz'
images_path = './Dataset/Data'


class Model(Chain):
    def __init__(self, out_size, chainermodel=chainermodel):
        super(Model, self).__init__(
            vgg=L.VGG16Layers(chainermodel),
            fc=L.Linear(None, out_size)
        )

    def __call__(self, x, train=True, extract_feature=False):
        with chainer.using_config('train', train):
            h = self.vgg(x, layers=['fc7'])['fc7']
            if extract_feature:
                return h
            y = self.fc(h)
        return y