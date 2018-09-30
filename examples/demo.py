import matplotlib.pyplot as plt
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import *
from keras.preprocessing import image
from keras.models import Sequential
from keras import backend as K
from keras.layers import *

from heatmap import synset_to_dfs_ids
from heatmap import to_heatmap


def display_heatmap(new_model, img_path, ids, preprocessing=None):
    # The quality is reduced.
    # If you have more than 8GB of RAM, you can try to increase it.
    img = image.load_img(img_path, target_size=(800, 1280))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if preprocessing is not None:
        x = preprocess_input(x)

    out = new_model.predict(x)

    heatmap = out[0]  # Removing batch axis.

    if K.image_data_format() == 'channels_first':
        heatmap = heatmap[ids]
        if heatmap.ndim == 3:
            heatmap = np.sum(heatmap, axis=0)
    else:
        heatmap = heatmap[:, :, ids]
        if heatmap.ndim == 3:
            heatmap = np.sum(heatmap, axis=2)

    plt.imshow(heatmap, interpolation="none")
    plt.show()


model = VGG16()
new_model = to_heatmap(model)

s = "n02084071"  # Imagenet code for "dog"
ids = synset_to_dfs_ids(s)
display_heatmap(new_model, "./dog.jpg", ids, preprocess_input)


# Also testing with a sequential model
sequential_model = Sequential()

sequential_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
sequential_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
sequential_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 2
sequential_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
sequential_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
sequential_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 3
sequential_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
sequential_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
sequential_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
sequential_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 5
sequential_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
sequential_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Classification block
sequential_model.add(Flatten())
sequential_model.add(Dense(1000, activation='relu'))
sequential_model.add(Dense(500, activation='relu'))
sequential_model.add(Dense(1000, activation='softmax'))

sequential_model.compile('sgd', loss='mse')
new_model = to_heatmap(sequential_model)

s = "n02084071"  # Imagenet code for "dog"
ids = synset_to_dfs_ids(s)
display_heatmap(new_model, "./dog.jpg", ids, preprocess_input)


# Also testing with a sequential model
sequential_model = Sequential()

for layer in model.layers:
    sequential_model.add(layer)
sequential_model.build(K.int_shape(model.input))

sequential_model.compile('sgd', loss='mse')
new_model = to_heatmap(sequential_model)

s = "n02084071"  # Imagenet code for "dog"
ids = synset_to_dfs_ids(s)
display_heatmap(new_model, "./dog.jpg", ids, preprocess_input)