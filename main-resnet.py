from __future__ import print_function
from matplotlib import pyplot as plt
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import keras

np.random.seed(1671)

print('################################################################################')
print()    

print('Reading train 60000.cdb ...')
X_train, Y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                    images_height=28,
                                    images_width=28,
                                    one_hot=False,
                                    reshape=True)

print('Reading test 20000.cdb ...')
X_test, Y_test = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                                images_height=28,
                                images_width=28,
                                one_hot=False,
                                reshape=True)

print('################################################################################')
print()    
print('Begin Deep Learning Process (Simple Deep Learning')


NB_EPOCH = 100
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
RESHAPE = 784
DROPOUT = 0.2
IMG_ROW, IMG_COL = 28, 28
INPUT_SHAPE = (IMG_ROW, IMG_COL, 1)

# K.set_image_dim_ordering("th")
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(20000, 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test  /= 255

print('{} Train Samples'.format(X_train.shape[0]))
print('{} Test Samples'.format(X_test.shape[0]))

Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

model = Sequential()
input_img = Input(INPUT_SHAPE, name='input_layer')
zeroPad1 = ZeroPadding2D((1,1), name='zeroPad1')
zeroPad1_2 = ZeroPadding2D((1,1), name='zeroPad1_2')
layer1 = Conv2D(6, (3, 3), strides=(2, 2), kernel_initializer='he_uniform', name='major_conv')
layer1_2 = Conv2D(16, (3, 3), strides=(2, 2), kernel_initializer='he_uniform', name='major_conv2')
zeroPad2 = ZeroPadding2D((1,1), name='zeroPad2')
zeroPad2_2 = ZeroPadding2D((1,1), name='zeroPad2_2')
layer2 = Conv2D(6, (3, 3), strides=(1,1), kernel_initializer='he_uniform', name='l1_conv')
layer2_2 = Conv2D(16, (3, 3), strides=(1,1), kernel_initializer='he_uniform', name='l1_conv2')
zeroPad3 = ZeroPadding2D((1,1), name='zeroPad3')
zeroPad3_2 = ZeroPadding2D((1,1), name='zeroPad3_2')
layer3 = Conv2D(6, (3, 3), strides=(1, 1), kernel_initializer='he_uniform', name='l2_conv')
layer3_2 = Conv2D(16, (3, 3), strides=(1, 1), kernel_initializer='he_uniform', name='l2_conv2')
layer4 = Dense(64, activation='relu', kernel_initializer='he_uniform', name='dense1')
layer5 = Dense(16, activation='relu', kernel_initializer='he_uniform', name='dense2')
final = Dense(10, activation='softmax', kernel_initializer='he_uniform', name='classifier')
first = zeroPad1(input_img)
second = layer1(first)
second = BatchNormalization(axis=-1)(second)
second = Activation('relu', name='major_act')(second)
third = zeroPad2(second)
third = layer2(third)
third = BatchNormalization(axis=-1)(third)
third = Activation('relu', name='l1_act')(third)
third = zeroPad3(third)
third = layer3(third)
third = BatchNormalization(axis=-1)(third)
third = Activation('relu', name='l1_act2')(third)
res =  keras.layers.Add()([third, second]) 
first2 = zeroPad1_2(res)
second2 = layer1_2(first2)
second2 = BatchNormalization(axis=-1)(second2)
second2 = Activation('relu', name='major_act2')(second2)
third2 = zeroPad2_2(second2)
third2 = layer2_2(third2)
third2 = BatchNormalization(axis=-1)(third2)
third2 = Activation('relu', name='l2_act')(third2)
third2 = zeroPad3_2(third2)
third2 = layer3_2(third2)
third2 = BatchNormalization(axis=-1)(third2)
third2 = Activation('relu', name='l2_act2')(third2)
res2 =  keras.layers.Add()([third2, second2]) 
res2 = Flatten()(res2)

res2 = layer4(res2)
res2 = Dropout(0.4, name='dropout1')(res2)
res2 = layer5(res2)
res2 = Dropout(0.4, name='dropout2')(res2)
res2 = final(res2)
model = Model(inputs=[input_img], outputs=[res2])
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)

print('Test Score:', score[0])
print('Test accuracy:', score[1])

