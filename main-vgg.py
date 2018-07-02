from __future__ import print_function
from matplotlib import pyplot as plt
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import keras

np.random.seed(1671)
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
INPUT_SHAPE = (IMG_ROW, IMG_COL, 1)

print('################################################################################')
print()    

print('Reading train 60000.cdb ...')
X_train, Y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                    images_height=IMG_ROW,
                                    images_width=IMG_COL,
                                    one_hot=False,
                                    reshape=True)

print('Reading test 20000.cdb ...')
X_test, Y_test = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                                images_height=IMG_ROW,
                                images_width=IMG_COL,
                                one_hot=False,
                                reshape=True)

print('################################################################################')
print()    
print('Begin Deep Learning Process (Simple Deep Learning')


X_train = X_train.reshape(60000, IMG_ROW, IMG_COL, 1)
X_test = X_test.reshape(20000, IMG_ROW, IMG_COL, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test  /= 255

print('{} Train Samples'.format(X_train.shape[0]))
print('{} Test Samples'.format(X_test.shape[0]))

Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMG_ROW, IMG_COL, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
model.add(Flatten())
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES, activation='relu'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print('Test Score:', score[0])
print('Test accuracy:', score[1])

