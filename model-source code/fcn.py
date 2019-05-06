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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ELU, GlobalAveragePooling2D
from keras.layers import LSTM, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt

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

# Prepare Training and Test dataset
X_train, y_train, train_labels = get_data(train_dir)
X_test, y_test, test_labels = get_data(test_dir)
num_categories = len(np.unique(y_train))
new_X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype('float32')
new_X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype('float32')
new_y_train = keras.utils.to_categorical(y_train, num_categories)
new_y_test = keras.utils.to_categorical(y_test, num_categories)

# Evaluate model
def evaluate_model(model, batch_size, epochs):
    checkpoint = keras.callbacks.ModelCheckpoint('fcn1weights{epoch:08d}.h5',save_weights_only=True, period=100)
    callbacks_list = [checkpoint]
    history = model.fit(new_X_train, new_y_train, batch_size=batch_size, epochs=epochs, verbose=1,callbacks=callbacks_list, validation_data=(new_X_test, new_y_test))
    score = model.evaluate(new_X_test, new_y_test, verbose=0)
    print('***Metrics Names***', model.metrics_names)
    print('***Metrics Values***', score)

# Build Model
convo = Sequential()
convo.add(Conv2D(32, (5, 5), strides=(1, 1), padding="same",activation="relu",input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3],), name='conv1'))
convo.add(BatchNormalization(axis=3, momentum=0.99, name='bn1'))
convo.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))

convo.add(Conv2D(48, (3, 3), strides=(1, 1), padding="same",activation="relu", name='conv2'))
convo.add(BatchNormalization(axis=3, momentum=0.99,name='bn2'))
convo.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

convo .add( Conv2D(64, (3, 3), strides=(1, 1), padding="same",activation="relu", name='conv3'))
convo .add(BatchNormalization(axis=3, momentum=0.99, name='bn3'))
convo .add(MaxPooling2D(pool_size=(2, 2), name='pool3'))


convo .add(Conv2D(48, (3, 3), strides=(1, 1), padding="same",activation="relu", name='conv4'))
convo .add(BatchNormalization(axis=3, momentum=0.99, name='bn4'))
convo .add(MaxPooling2D(pool_size=(2, 2), name='pool4'))

convo.add(Conv2D(32, (3, 3), strides=(1, 1), padding="same",activation="relu", name='conv5'))
convo .add( BatchNormalization(axis=3, momentum=0.99, name='bn5'))
convo .add(MaxPooling2D(pool_size=(2, 2), name='pool5'))

convo .add(Dropout(0.5))
convo .add(Conv2D(num_categories,1,1,activation='softmax',name='conv6'))
convo.add(GlobalAveragePooling2D())
convo.summary()
convo.compile(loss="categorical_crossentropy", optimizer=Adam(lr=.0001), metrics=['accuracy'])
#evaluate_model(convo, 2, 100)

# Save Model to disk
model_json = convo.to_json()
with open("fcn1_model1.json", "w") as json_file:
    json_file.write(model_json)
convo.save_weights("fcn1_model1.h5")
print("FCN model saved to disk")
print("FCN model saved to disk")

# Load model from disk
json_file = open('fcn1_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("fcn1weights00000100.h5")
plot_model(loaded_model, to_file='FCN1.png')
print("FCN model loaded from disk")

y_pred = loaded_model.predict(new_X_test, batch_size=None, verbose=0, steps=None).argmax(axis=-1)
res_crosstab = pd.crosstab(y_pred, y_test)
dict_idx_fruit = {idx: label for idx, label in enumerate(test_labels)}
print(dict_idx_fruit)
res_crosstab

# Test
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
    
    img_filename="Banana.jpg"
    current_img = io.imread(img_filename)
    current_img = transform.resize(current_img, (img_size, img_size))
    current_img = np.asarray(current_img)
    current_img = np.asarray([current_img])
layer_outputs = [layer.output for layer in loaded_model.layers[:12]] # Extracts the outputs of the top 12 layers
activation_model = loaded_model.Model(inputs=loaded_model.input, outputs=layer_outputs)
images_per_row = 16
layer_names=[]
for layer in loaded_model.layers[:12]:
    layer_names.append(layer.name)
    activations = activation_model.predict(current_img)
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
