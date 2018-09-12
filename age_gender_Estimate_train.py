import os.path,sys
sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/../../')
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks
import scipy.ndimage
import numpy as np
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers

nb_classes  = 20
base_dir = '.'
prefix = 'vgg16'
n_train_samples = 21027
train_file_name = 'bottleneck_features_train.npy'

n_validation_samples = 1000
validation_file_name = 'bottleneck_features_validation.npy'

# VGG16(model & weight)をインポート
model = VGG16(include_top=False, weights='imagenet')
model.summary()

# 画像データをnumpy arrayに変換
## training dataの読み込み
image_data_generator = ImageDataGenerator(rescale=1.0/255)
train_data = image_data_generator.flow_from_directory(
    'Dataset/traindat',
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

## validation dataの読み込み
image_data_generator = ImageDataGenerator(rescale=1.0/255)
validation_data = image_data_generator.flow_from_directory(
    'Dataset/testdat',
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

# VGG16を使用してボトルネック特徴量データを生成する
## training data
bottleneck_feature_train = model.predict_generator(train_data, n_train_samples, verbose=1)

## validation data
bottleneck_feature_validation = model.predict_generator(validation_data, n_validation_samples, verbose=1)

# bottleneck featuresの保存
## traning data
np.save(base_dir + prefix + train_file_name, bottleneck_feature_train)

## validation data
np.save(base_dir + prefix + validation_file_name, bottleneck_feature_validation)

# Bottleneck featuresの読み込み
train_data  = np.load(base_dir + prefix + train_file_name)
len_input_samples = len(train_data)
train_labels = np.array([0] * int(len_input_samples/2) + [1] * int(len_input_samples / 2))

validation_data = np.load(base_dir + prefix + validation_file_name)
validation_labels = np.array([0] * int(n_validation_samples / 2 *32) + [1] * int(n_validation_samples / 2 * 32))

input_shape = train_data.shape[1:]

model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

result_dir = 'history_vgg16_transfer_learning.txt'

# callbacks
callbacks = keras.callbacks.TensorBoard(log_dir='tensorBoard', histogram_freq=0)

# train model
history = model.fit(train_data, train_labels, epochs=20, batch_size=32, callbacks=[callbacks], validation_data=(validation_data, validation_labels))

# Save weight
model.save_weights('vgg16_transferlearning_weights.h5')

# Save history
loss = history.history['loss']
acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']