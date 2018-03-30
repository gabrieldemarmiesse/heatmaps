# heatmaps

### This repository contain functions that transform a functionnal keras model that classifies images into an heatmap generator.

Some code and a lot of great ideas come from this 
repository: https://github.com/heuritech/convnets-keras

The heatmaps resolution are quite limited (but still better 
than in the heuritech repository if the model has a flatten layer).

This code should work with Theano, Tensorflow, CNTK and with all data formats,
but it was only tested with Tensorflow.

### Installation

Now installable with pip!

```
git clone https://github.com/gabrieldemarmiesse/heatmaps.git
cd heatmaps
pip install -e .
```


### Example with VGG16
Here is a sample of code to understand what is going on:

```python
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras import backend as K

from heatmap import to_heatmap, synset_to_dfs_ids


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
```
<img src=https://raw.githubusercontent.com/gabrieldemarmiesse/heatmaps/master/examples/dog.jpg width="400px">

<img src=https://raw.githubusercontent.com/gabrieldemarmiesse/heatmaps/master/examples/heatmap_dog_vgg16.png width="400px">


The function to_heatmap also take a second argument: input_shape

This should be used only if your classifier doesn't have fixed sizes for width and height.
You must then give the image size that was used during training.
eg: to_heatmap(model, input_shape=(3,256,256))

### Example with ResNet50
You can also try this:
```python
from keras.applications.resnet50 import ResNet50, preprocess_input
from heatmap import to_heatmap, synset_to_dfs_ids
model = ResNet50()
new_model = to_heatmap(model)

s = "n02084071"  # Imagenet code for "dog"
ids = synset_to_dfs_ids(s)
display_heatmap(new_model, "./dog.jpg", ids, preprocess_input)
```
<img src=https://raw.githubusercontent.com/gabrieldemarmiesse/heatmaps/master/examples/dog.jpg width="400px">

<img src=https://raw.githubusercontent.com/gabrieldemarmiesse/heatmaps/master/examples/heatmap_dog_resnet.png width="400px">


### Example with your own model
It should also work with custom classifiers. 
Let's say your classifier has two classes: dog (first class) and not dog (second class).
Then this code can get you a heatmap:

```python
new_model = to_heatmap(my_custom_model)
idx = 0  # The index of the class you care about, here the first one.
display_heatmap(new_model, "./dog.jpg", idx)
```

### Note on the sizes of the heatmaps

Due to the topology of common classification neural networks,
the heatmap produced will be smaller than the input image.
The downsampling usually happen at maxpool layers or at strided convolution
layers.

Here is a table to get an idea of the size of the heatmap that you will obtain.

The size of the input image is assumed to be 1024x1024.

| Network        | Heatmap size for a 1024 x 1024 image |
| ------------- |:-----------------------------------:|
| VGG16      | 51 x 51                      |
| VGG19 | 51 x 51                           |
| ResNet50      | 26 x 26                      |
| InceptionV3 | 30 x 30                         |
| Xception      | 32 x 32                     |
| InceptionResnetV2 | 30 x 30                           |
| MobileNet      | 32 x 32                     |
| DenseNet121 | 32 x 32                           |

The VGG16 and 19 have a better resolution because we can use a trick 
before the flatten layer, and replace the convolutions by dilated convolutions.

This library performs this optimization out of the box, without you having to do anything.
