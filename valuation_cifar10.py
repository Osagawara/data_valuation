import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential


PEOPLE = 100
CATEGORY = 10
BATCH_SIZE = 50
INPUT_SHAPE = [32, 32, 3]


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
