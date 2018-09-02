import chainer
import chainercv
import caffe
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import cupy as xp

mean_filename = '/home/ryuzo/Downloads/age_gender_mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean = caffe.io.blobproto_to_array(a)[0]
age_net_pretrained = '/home/ryuzo/Downloads/age_net.caffemodel'
age_net_model_file = '/home/ryuzo/Downloads/deploy_age.prototxt'
age_net = caffe.Classifier(age_net_model_file, age_net_pretrained, mean=mean, channel_swap=(2, 1, 0), raw_scale=255, image_dims=(256, 256))
gender_net_pretrained = '/home/ryuzo/Downloads/gender_net.caffemodel'
gender_net_model_file = '/home/ryuzo/Downloads/deploy_gender.prototxt'
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained, mean=mean, channel_swap=(2, 1, 0), raw_scale=255, image_dims=(256, 256))
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']

example_image = '/home/ryuzo/Pictures/face_sample.jpg'
input_image = caffe.io.load_image(example_image)
age_prediction = age_net.predict([input_image])
gender_prediction = gender_net.predict([input_image])
age_list[age_prediction[0].argmax()]
gender_list[gender_prediction[0].argmax()]

