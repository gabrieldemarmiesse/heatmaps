from keras.applications import *
from helper import helper_test


def test_vgg16():
    helper_test(VGG16())


def test_resnet50():
    helper_test(ResNet50())


if __name__ == '__main__':
    test_vgg16()
    test_resnet50()