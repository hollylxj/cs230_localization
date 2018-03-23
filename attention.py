from keras.applications import VGG16
from vis.utils import utils
from keras import activations

# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

#%autoreload
from keras.preprocessing import image
from keras import applications
from keras.models import Sequential
from keras.applications import vgg16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Conv2D, Conv3D,Input, ZeroPadding3D, Reshape
from keras.layers.convolutional import Convolution2D, Convolution3D, MaxPooling2D, ZeroPadding2D,ZeroPadding3D 
from keras.layers.core import Reshape
import os
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.layers import Merge 
import numpy as np
import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import random

model_right = Sequential()
# input_x = Input((299,299,10,3))
# model_right.add(input_x)
model_right.add(Conv3D(3,(3,3,10), activation='relu', data_format="channels_last", input_shape=((299,299,10,3))))
model_right.add(ZeroPadding3D((1,1,0)))
model_right.add(Reshape((299,299,3)))
depth_model_vgg16 = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(299,299,3))
model_right.add(depth_model_vgg16)
model_right.add(Flatten())
model_right.add(Dense(1024, activation='relu'))
model_right.add(Dropout(0.5))
model_right.add(Dense(512, activation='relu'))
model_right.add(Dropout(0.5))
model_right.add(Dense(3, activation = 'relu'))  
model_right.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=1e-4),
            metrics=['mean_squared_error'])
model_right.summary()

model_right.load_weights("/Users/zhouzixuan/Documents/DL/result/weights.best.hdf5")
layer_idx = utils.find_layer_idx(model_right, 'dense_3')

# Swap softmax with linear
model_right.layers[layer_idx].activation = activations.linear
model_right = utils.apply_modifications(model_right)

from vis.utils import utils
from matplotlib import pyplot as plt
import cv2
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations

plt.rcParams['figure.figsize'] = (18, 6)
layer_idx = utils.find_layer_idx(model_right, 'dense_3')
rgb_img_in = cv2.imread('./a.jpg', 1)
depth_img_in = cv2.imread('./a_d.jpg', 0)

rgb_mean = np.mean(rgb_img_in)
rgb_std = np.std(rgb_img_in) ** 2
rgb_img = np.array([(rgb_img_in[ii]-rgb_mean)/rgb_std for ii in range(len(rgb_img_in))]) 
        
depth_mean = np.mean(depth_img_in)
depth_std = np.std(depth_img_in) ** 2
depth_img = np.array([(depth_img_in[ii]-depth_mean)/depth_std for ii in range(len(depth_img_in))]) 

print rgb_img.shape
print depth_img.shape

import tensorflow as tf
d_batch = depth_img
x_batch = rgb_img 
d_round = np.floor(d_batch/25.5)
sess = tf.InteractiveSession()
v = tf.one_hot(d_round, depth=10, axis=2, on_value=1.0, off_value=0.0)
v = v.eval()
print v.shape
combine = np.empty([299, 299, 3, 0])
for i in range(10):
    v_tmp = v[:,:,i]
    v_tmp = np.expand_dims(v_tmp,axis=2)
    v_tmp = np.broadcast_to(v_tmp,(299, 299, 3))
    v_tmp = v_tmp == 1

    x_tmp = np.multiply(x_batch, v_tmp)
    x_cur = np.expand_dims(x_tmp, axis = 3)
    combine = np.concatenate((combine, x_cur), axis=3)

combine = np.transpose(combine, (0,1,3,2))
print combine.shape

# grad1 = visualize_saliency(model_right, layer_idx, filter_indices=0, seed_input= combine, backprop_modifier='relu')
grads = visualize_saliency(model_right, layer_idx, filter_indices=0, seed_input= combine, backprop_modifier='relu' )
f, ax = plt.subplots(1)
ax.imshow(grads[:,:,9,:])

















