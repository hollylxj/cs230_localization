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
#import imp
#imp.reload(parse_data)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
   # return np.linalg.norm(y_pred-y_true)/np.linalg.norm(y_true)
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


random.seed(666)
index_map = range(102)
random.shuffle(index_map)
print index_map

object_path = "/home/zhouzixuan/notebooks/proj_new/3d_data/"
dev_path = "/home/zhouzixuan/notebooks/proj_new/3d_data/"
dev_num = 10
X_dev = np.empty([0,299,299,10,3])
y_dev = np.empty([0,3])
rescale  = 1./255
for j in range(dev_num):
    k = index_map[j + 90]
    x_dev = np.load(dev_path+str(k)+".npy") * rescale 
    x_dev = np.transpose(x_dev, (0,1,2,4,3))
    X_dev = np.vstack([X_dev, x_dev])
    tmp2 = np.load(dev_path+str(k)+"_y.npy")
    y_dev = np.vstack([y_dev, tmp2])
print y_dev.shape
print X_dev.shape

def train(model):
    filename = "vgg3d_"
    fh = open(filename + 'report.txt','w')
    csv_logger = CSVLogger('logvgg.csv', append=True, separator=';') 
    f_train = open(filename + 'train_report.txt',"a", 0)
    f_train_step = open(filename + 'step_report.txt',"a", 0)
    f_dev = open(filename + 'dev_report.txt',"a", 0)
    batch = 32
    epochs = 20
    batch_num = 85
    rescale=1. / 255
    filepath_best="/home/zhouzixuan/notebooks/proj_new/3d_data/weights.best.hdf5"
    if os.path.exists(filepath_best):
        model.load_weights(filepath_best)
        print "load weight success!"
    for e in range(epochs):
        print('Epoch', e)
        for p in range(batch_num):
            b = index_map[p]
            x_batch = np.load(object_path+str(b)+".npy")
            x_batch = np.transpose(x_batch, (0,1,2,4,3))        
            y_batch = np.load(object_path+str(b)+"_y.npy")        
            checkpoint = ModelCheckpoint(filepath=filepath_best,monitor='loss', verbose=1,save_best_only=True, mode='min')
            res = model.fit(x_batch, y_batch,callbacks=[checkpoint],verbose=0)
            print str(res.history)
            f_train_step.write(str(res.history))
            f_train_step.write("\n")
        train_loss = model.evaluate(x_batch, y = y_batch)
        f_train.write(str(train_loss))
        dev_loss = model.evaluate(x = X_dev, y = y_dev)
        f_dev.write(str(dev_loss))
        print "----val_loss & l2_error----:"
        print dev_loss
    f_train.close()
    f_train_step.close()
    f_dev.close()

train(model_right)    








