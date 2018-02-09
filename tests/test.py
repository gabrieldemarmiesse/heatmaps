import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

from heatmap import synset_to_dfs_ids
from heatmap import to_heatmap

img_path = "../examples/dog.jpg"
model = VGG16()
new_model = to_heatmap(model)

# Loading the image
img = image.load_img(img_path, target_size=(800, 800))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

out = new_model.predict(x)

s = "n02084071"  # Imagenet code for "dog"
ids = synset_to_dfs_ids(s)
heatmap = out[0]
if K.image_data_format() == 'channels_first':
    heatmap = heatmap[ids]
    heatmap = np.sum(heatmap, axis=0)
else:
    heatmap = heatmap[:, :, ids]
    heatmap = np.sum(heatmap, axis=2)
print(heatmap.shape)
assert heatmap.shape[0] == heatmap.shape[1]
