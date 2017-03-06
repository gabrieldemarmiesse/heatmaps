# heatmaps

### This repository contain functions that transform a functionnal keras model that classifies images into an heatmap generator.

Some code and a lot of great ideas come from this repository: https://github.com/heuritech/convnets-keras

For the moment, it only works with Theano and the heatmaps resolution are quite limited (but still better than in the heuritech repository if the model has a flatten layer).

Here is a sample of code to understand what is going on:

```python
from heatmap import to_heatmap
from heatmap import synset_to_dfs_ids
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from keras.applications.vgg16 import VGG16

def display_heatmap(new_model, img_path):
    
    # The quality is reduced. 
    # If you have more than 8GB of RAM, you can try to increase it.
    img = image.load_img(img_path, target_size=(800, 1280)) 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    out = new_model.predict(x)

    s = "n02084071" # Imagenet code for "dog"
    ids = synset_to_dfs_ids(s)
    heatmap_dog = out[0,ids].sum(axis=0)
    plt.imshow(heatmap_dog, interpolation="none")
    plt.show()
    
model = VGG16()
new_model = to_heatmap(model)
display_heatmap(new_model, "./dog.jpg")

```
<img src=https://raw.githubusercontent.com/gabrieldemarmiesse/heatmaps/master/examples/dog.jpg width="400px">

<img src=https://raw.githubusercontent.com/gabrieldemarmiesse/heatmaps/master/examples/heatmap_dog_vgg16.png width="400px">


The function to_heatmap also take a second argument: input_shape

This should be used only if your classifier doesn't have a fixed size.
You must then give the image size that was used during training.
eg: to_heatmap(model, input_shape=(3,256,256))


You can also try this:
```python
model = ResNet50()
new_model = to_heatmap(model)
display_heatmap(new_model, "./dog.jpg")
```
<img src=https://raw.githubusercontent.com/gabrieldemarmiesse/heatmaps/master/examples/dog.jpg width="400px">

<img src=https://raw.githubusercontent.com/gabrieldemarmiesse/heatmaps/master/examples/heatmap_dog_resnet.png width="400px">

It should also work with custom classifiers. 
Let's say your classifier has two classes: dog (first class) and not dog (second class).
Then this code can get you a heatmap:

```python
out = new_model.predict(x)
heatmap_dog = out[0,[0].sum(axis=0) # We select only the result for the first class.
plt.imshow(heatmap_dog, interpolation="none")
plt.show()
```
