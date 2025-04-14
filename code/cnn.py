from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Recall
import pyarrow.feather as feather
import numpy as np

def fish_cnn1(img_rows, img_cols):
    """
    creates CNN for classifying fish images
    params:
        img_rows (int): number of rows in the image
        img_cols (int): number of cols in the image
    returns: CNN model
    """
    inpx = Input(shape=(img_rows, img_cols, 1))

    # can also add padding in here with the padding = 'valid' (no padding) or 'same' (padding) option
    # check out the full range of options here: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    conv1 = Conv2D(1, kernel_size=(3, 3), strides=1, activation=None, padding='same')(inpx)
    # pool_layer = MaxPooling2D(pool_size=(3, 3))(conv1)

    conv2 = Conv2D(1, kernel_size=(3, 3), strides=1, activation=None)(conv1)

    # need to flatten the images for the hidden layer/output
    flat_G = Flatten()(conv2)

    # can decide how many hidden nodes to use
    hidden1 = Dense(1000, activation='linear')(flat_G)
    hidden2 = Dense(500, activation='relu')(hidden1)

    out_layer = Dense(1, activation='sigmoid')(hidden2)

    model = Model([inpx], [out_layer])
    model.compile(optimizer=SGD(),
                loss=binary_crossentropy,
                metrics=['accuracy'])
    
    return model

def fish_cnn2(img_rows, img_cols):
    """
    creates CNN for classifying fish images
    params:
        img_rows (int): number of rows in the image
        img_cols (int): number of cols in the image
    returns: CNN model
    """
    inpx = Input(shape=(img_rows, img_cols, 1))

    conv1 = Conv2D(1, kernel_size=(3, 3), strides=1, activation=None, padding='same')(inpx)
    conv2 = Conv2D(1, kernel_size=(3, 3), strides=1, activation=None)(conv1)

    # need to flatten the images for the hidden layer/output
    flat_G = Flatten()(conv2)

    # can decide how many hidden nodes to use
    hidden1 = Dense(1000, activation='sigmoid')(flat_G)
    hidden2 = Dense(500, activation='sigmoid')(hidden1)

    out_layer = Dense(1, activation='sigmoid')(hidden2)

    model = Model([inpx], [out_layer])
    model.compile(optimizer=SGD(),
                loss=binary_crossentropy,
                metrics=['accuracy'])
    
    return model

def fish_cnn3(img_rows, img_cols):
    """
    creates CNN for classifying fish images
    params:
        img_rows (int): number of rows in the image
        img_cols (int): number of cols in the image
    returns: CNN model
    """
    inpx = Input(shape=(img_rows, img_cols, 1))

    conv1 = Conv2D(1, kernel_size=(3, 3), strides=1, activation=None, padding='same')(inpx)
    conv2 = Conv2D(1, kernel_size=(3, 3), strides=1, activation=None)(conv1)

    # need to flatten the images for the hidden layer/output
    flat_G = Flatten()(conv2)

    # can decide how many hidden nodes to use
    hidden1 = Dense(1000, activation='relu')(flat_G)
    hidden2 = Dense(500, activation='relu')(hidden1)

    out_layer = Dense(1, activation='sigmoid')(hidden2)

    model = Model([inpx], [out_layer])
    model.compile(optimizer=SGD(),
                loss=binary_crossentropy,
                metrics=['accuracy'])
    
    return model

def fish_cnn4(img_rows, img_cols):
    """
    creates CNN for classifying fish images
    params:
        img_rows (int): number of rows in the image
        img_cols (int): number of cols in the image
    returns: CNN model
    """
    # TODO: optimize
    inpx = Input(shape=(img_rows, img_cols, 1))

    conv1 = Conv2D(1, kernel_size=(3, 3), strides=1, activation=None, padding='same')(inpx)
    conv2 = Conv2D(1, kernel_size=(3, 3), strides=1, activation=None)(conv1)

    # need to flatten the images for the hidden layer/output
    flat_G = Flatten()(conv2)

    # can decide how many hidden nodes to use
    hidden1 = Dense(1000, activation='sigmoid')(flat_G)
    hidden2 = Dense(500, activation='relu')(hidden1)
    hidden3 = Dense(250, activation='relu')(hidden2)

    out_layer = Dense(1, activation='sigmoid')(hidden3)

    model = Model([inpx], [out_layer])
    model.compile(optimizer=SGD(),
                loss=binary_crossentropy,
                metrics=['accuracy'])
    
    return model
