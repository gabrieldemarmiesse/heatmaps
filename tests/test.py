from keras.applications import *
from helper import helper_test


def test_vgg16():
    helper_test(VGG16())


def test_resnet50():
    helper_test(ResNet50())


def test_inception_v3():
    helper_test(InceptionV3())


def test_densenet():
    helper_test(DenseNet121())


def test_mobilenet():
    helper_test(MobileNet())



if __name__ == '__main__':
    test_vgg16()
    test_resnet50()
    test_inception_v3()
    test_densenet()
    #test_mobilenet()