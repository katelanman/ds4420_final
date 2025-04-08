from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
import pyarrow.feather as feather

def fish_cnn(img_rows, img_cols):
    """
    creates CNN for classifying fish images
    params:
        img_rows (int): number of rows in the image
        img_cols (int): number of cols in the image
    returns: CNN model
    """
    # TODO: optimize
    inpx = Input(shape=(img_rows, img_cols, 1))

    # can also add padding in here with the padding = 'valid' (no padding) or 'same' (padding) option
    # check out the full range of options here: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    conv1 = Conv2D(1, kernel_size=(3, 3), strides=1, activation=None, padding='same')(inpx)
    # pool_layer = MaxPooling2D(pool_size=(3, 3))(conv1)

    conv2 = Conv2D(1, kernel_size=(3, 3), strides=1, activation=None)(conv1)

    # need to flatten the images for the hidden layer/output
    flat_G = Flatten()(conv2)

    # can decide how many hidden nodes to use
    hidden1 = Dense(250, activation='linear')(flat_G)
    hidden2 = Dense(250, activation='relu')(hidden1)

    out_layer = Dense(1, activation='sigmoid')(hidden2)

    model = Model([inpx], [out_layer])
    model.compile(optimizer=SGD(),
                loss=binary_crossentropy,
                metrics=['accuracy'])
    
    return model

# get data
data = feather.read_feather("data/working/fish_frames.feather")
X = data.drop('label', axis=1).to_numpy()
y = data['label'].to_numpy()

print('data read')

img_rows, img_cols = 288, 352

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(y_test)

# add dimensionality and scale
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1) / 255
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1) / 255

print('data split and scaled')

model = fish_cnn(img_rows, img_cols)
model.fit(X_train, y_train, epochs=10, verbose=True, batch_size=10)

print('model fit')

score = model.evaluate(X_test, y_test, verbose=0)
print('loss=', score[0])
print('accuracy=', score[1])