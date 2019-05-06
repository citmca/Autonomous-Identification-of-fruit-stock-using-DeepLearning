import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import skimage
from skimage import io, transform
from IPython.display import Image, display
import tensorflow as tf
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import LSTM, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
img_size = 100
train_dir = 'X:/project/fruits-360/Training/'
test_dir = 'X:/project/fruits-360/Test/'
prdict = 'X:/project/fruits-360/Predict/'


def get_data(folder_path):
    imgs = []
    indices = []
    labels = []
    for idx, folder_name in enumerate(os.listdir(folder_path)[:35]):
        if not folder_name.startswith('.'):
            labels.append(folder_name)
            for file_name in tqdm(os.listdir(folder_path + folder_name)):
                if not file_name.startswith('.'):
                    img_file = io.imread(folder_path + folder_name + '/' + file_name)
                    if img_file is not None:
                        img_file = transform.resize(img_file, (img_size, img_size))
                        imgs.append(np.asarray(img_file))
                        indices.append(idx)
    imgs = np.asarray(imgs)
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    return imgs, indices, labels


X_train, y_train, train_labels = get_data(train_dir)


X_test, y_test, test_labels = get_data(test_dir)
num_categories = len(np.unique(y_train))
new_X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype('float32')
new_X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype('float32')
new_y_train = keras.utils.to_categorical(y_train, num_categories)
new_y_test = keras.utils.to_categorical(y_test, num_categories)


def evaluate_model(model, batch_size, epochs):

    checkpoint = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5',save_weights_only=True, period=1000)
    callbacks_list = [checkpoint]
    history = model.fit(new_X_train, new_y_train, batch_size=batch_size, epochs=epochs, verbose=1,callbacks=callbacks_list, validation_data=(new_X_test, new_y_test))
    score = model.evaluate(new_X_test, new_y_test, verbose=0)
    print('***Metrics Names***', model.metrics_names)
    print('***Metrics Values***', score)


convolutional = Sequential()

convolutional.add(Conv2D(5, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3],)))
convolutional.add(Conv2D(3, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))


convolutional.add(Conv2D(3, kernel_size=(3,3), activation='relu'))
convolutional.add(Conv2D(1, (3, 3), activation='relu'))
convolutional.add(MaxPooling2D(pool_size=(2, 2)))
convolutional.add(Dropout(0.35))

convolutional.add(Flatten())

convolutional.add(Dense(512, activation='relu'))
convolutional.add(Dropout(0.6))
convolutional.add(BatchNormalization())
convolutional.add(Dense(num_categories, activation='softmax'))

convolutional.summary()
convolutional.compile(loss="categorical_crossentropy", optimizer=Adam(lr=.0001), metrics=['accuracy'])

#evaluate_model(convolutional, 4, 10000)

model_json = convolutional.to_json()
with open("cnn_model1.json", "w") as json_file:
    json_file.write(model_json)
convolutional.save_weights("cnn_model1.h5")

print("CNN model saved to disk")
# Load model from disk
json_file = open('cnn_model1.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("weights00006000.h5")
plot_model(loaded_model, to_file='CNN.png')
print("CNN model loaded from disk")
y_pred = loaded_model.predict(new_X_test, batch_size=None, verbose=0, steps=None).argmax(axis=-1)
res_crosstab = pd.crosstab(y_pred, y_test)

dict_idx_fruit = {idx: label for idx, label in enumerate(test_labels)}
print(dict_idx_fruit)

res_crosstab

for idx in range(num_categories):
    accuracy = res_crosstab.loc[idx, idx] / res_crosstab.loc[:, idx].sum()
    flag = '***LOW***' if accuracy < 0.75 else ''
    print(dict_idx_fruit[idx])
    print('   ', flag, 'accuracy –', round(accuracy * 100, 2), '%')


def get_one_img_per_fruit(folder_path):
    printouts = []
    for idx, folder_name in enumerate(os.listdir(folder_path)[:35]):
        if not folder_name.startswith('.'):
            for idx2, file_name in enumerate(tqdm(os.listdir(folder_path + folder_name))):
                if idx2 == 0:
                    if not file_name.startswith('.'):
                        img_filename = folder_path + folder_name + '/' + file_name
                        ig = Image(filename=img_filename)
                        display(ig)
                        current_img = io.imread(img_filename)
                        current_img = transform.resize(current_img, (img_size, img_size))
                        current_img = np.asarray(current_img)
                        current_img = np.asarray([current_img])

                        current_pred = loaded_model.predict(current_img, batch_size=None, verbose=0, steps=None).argmax(
                            axis=-1)
                        current_pred = dict_idx_fruit[current_pred[0]]

                        is_incorrect = 'INCORRECT' if folder_name != current_pred else ''

                        msg = '{} – predicted as {} {}'.format(folder_name, current_pred, is_incorrect)
                        print(msg)
                        printouts.append(msg)
    return printouts

print("predict")
printouts = get_one_img_per_fruit(prdict)



for msg in printouts:
    print(msg)

