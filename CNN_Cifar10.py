import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adagrad
from keras.datasets import cifar10
from keras.utils import to_categorical


class Cifar10_Classifier():

    def __init__(self, input_shape, num_class, regu_weight):
        self.input_shape = input_shape
        self.num_class = num_class
        self.regu_weight = regu_weight

    def construct(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.input_shape))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu', kernel_regularizer=l2(self.regu_weight)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_class, activation='softmax', kernel_regularizer=l2(self.regu_weight)))

    def get_classifier(self):
        return self.model





if __name__ == '__main__':
    input_shape = [32, 32, 3]
    num_class = 10
    learning_rate = 0.0001
    decay = 0.0005
    batch_size = 32
    epochs = 20

    opt = Adagrad(lr=learning_rate, decay=decay)
    classifier = Cifar10_Classifier(input_shape=input_shape, num_class=num_class, regu_weight=0.01)
    classifier.construct()

    model = classifier.get_classifier()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.metrics_names)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(y_train)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, num_class)
    y_test = to_categorical(y_test, num_class)

    a = [1, 2, 3]
    b = np.random.permutation(3)
    print(a[b])

    # model.fit(x_train, y_train, batch_size, epochs, validation_data=(x_test, y_test), shuffle=True)


