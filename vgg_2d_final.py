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
import numpy as np
import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint
#import imp
#imp.reload(parse_data)
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
   # return np.linalg.norm(y_pred-y_true)/np.linalg.norm(y_true)

model_vgg16 = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(299,299,3))
input_x = Input((299,299,4))
x = Conv2D(3,(3,3), activation='relu', data_format="channels_last")(input_x)
x = ZeroPadding2D((1,1))(x)
x = Reshape((299,299,3))(x)
x = model_vgg16(x)

flatten = Flatten()
new_layer2 = Dense(1024, activation='relu', name='my_dense_2')
out2 = new_layer2(flatten(x))
x = Dropout(0.5)(out2)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(3, activation = 'relu')(x)

model_start = Model(input_x, preds)
model_start.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=1e-4),
            metrics=['mean_squared_error'])
model_start.summary()

object_path = "/home/zhouzixuan/notebooks/proj_new/3d_data/"
dev_path = "/home/zhouzixuan/notebooks/proj_new/3d_data/"
filename ="vgg"
img_width, img_height = 299, 299

dev_num = 10
X_dev = np.empty([0,299,299,4])
y_dev = np.empty([0,3])
for k in range(dev_num):
    x_dev = np.load(dev_path+str(90+k)+"_x.npy")
    d_dev = np.load(dev_path+str(90+k)+"_d.npy")
    tmp1 = (np.concatenate((x_dev,np.expand_dims(d_dev, axis=3)), axis=3)) 
    X_dev = np.vstack([X_dev, tmp1])
    tmp2 = np.load(dev_path+str(90+k)+"_y.npy")
    y_dev = np.vstack([y_dev, tmp2])
print y_dev.shape
print X_dev.shape

def train(model):
    filename = "vgg2d_"
    fh = open(filename + 'report.txt','w')
    csv_logger = CSVLogger('logvgg.csv', append=True, separator=';') 
    f_train = open(filename + 'train_report.txt',"a", 0)
    f_train_step = open(filename + 'step_report.txt',"a", 0)
    f_dev = open(filename + 'dev_report.txt',"a", 0)
    batch = 32
    epochs = 10
    batch_num = 90
    rescale=1. / 255
    filepath_best="/home/zhouzixuan/proj/data/weights.best.hdf5"
    if os.path.exists(filepath_best):
        model.load_weights(filepath_best)
        print "load weight success!"
    for e in range(epochs):
        print('Epoch', e)
        for b in range(batch_num):
            x_batch = np.load(object_path+str(b)+"_x.npy")
            d_batch = np.load(object_path+str(b)+"_d.npy")
            X = (np.concatenate((x_batch,np.expand_dims(d_batch, axis=3)), axis=3))           
            y_batch = np.load(object_path+str(b)+"_y.npy")            
            checkpoint = ModelCheckpoint(filepath=filepath_best,monitor='loss', verbose=1,save_best_only=True, mode='min')
            res = model.fit(X, y_batch,callbacks=[checkpoint],verbose=0)
            print str(res.history)
            f_train_step.write(str(res.history))
            f_train_step.write("\n")
        train_loss = model.evaluate(x = X, y = y_batch)
        f_train.write(str(train_loss))
        dev_loss = model.evaluate(x = X_dev, y = y_dev)
        f_dev.write(str(dev_loss))
        print "----val_loss & l2_error----:"
        print dev_loss
    f_train.close()
    f_train_step.close()
    f_dev.close()

train(model_start)



