from __future__ import print_function
from matplotlib import pyplot as plt
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

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


NB_EPOCH = 10
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
RESHAPE = 784
DROPOUT = 0.2
IMG_ROW, IMG_COL = 28, 28
INPUT_SHAPE = (1, IMG_ROW, IMG_COL)

K.set_image_dim_ordering("th")
X_train = X_train.reshape(60000, 1, 28, 28)
X_test = X_test.reshape(20000, 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test  /= 255

print('{} Train Samples'.format(X_train.shape[0]))
print('{} Test Samples'.format(X_test.shape[0]))

Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)

print('Test Score:', score[0])
print('Test accuracy:', score[1])

