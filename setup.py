from setuptools import setup
from setuptools import find_packages


setup(name='Keras to heatmaps',
      version='0.2',
      description='Make heatmaps generators from keras models',
      author='Gabriel de Marmiesse',
      install_requires=['numpy',
                        'keras>=2.2.2',
                        'scipy',
                        'pytest'],
      packages=find_packages(),
      include_package_data=True)
