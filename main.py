from __future__ import print_function
from matplotlib import pyplot as plt
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np

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

X_train = X_train.reshape(60000, RESHAPE)
X_test = X_test.reshape(20000, RESHAPE)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test  /= 255

print('{} Train Samples'.format(X_train.shape[0]))
print('{} Test Samples'.format(X_test.shape[0]))

Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPE,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)

print('Test Score:', score[0])
print('Test accuracy:', score[1])