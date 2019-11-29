import os
import time
import numpy as np
from tqdm import tqdm
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adagrad
from CNN_Cifar10 import Cifar10_Classifier

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def is_convergent(original: np.ndarray, new: np.ndarray, criterion = 0.0001):
    assert len(original) == len(new)
    max_diff = np.max(np.abs(original - new))
    return max_diff, max_diff < criterion

def data_valuation(data: list, label: list, constructor: Cifar10_Classifier):
    population = len(data)
    new_value = np.zeros(population)
    opt = Adagrad(lr=learning_rate)
    times = 0

    try:
        pbar = tqdm(total=10000)
        time.sleep(5)
        print()
        while True:
            times += 1
            old_value = np.copy(new_value)
            old_acc = 0

            constructor.construct()
            model = constructor.get_classifier()
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            for i in np.random.permutation(population):
                model.train_on_batch(data[i], label[i])
                _, new_acc = model.evaluate(x_test, y_test, verbose=0)
                new_value[i] = (times - 1) / times * old_value[i] + 1 / times * (new_acc - old_acc)
                old_acc = new_acc

            max_diff, tag = is_convergent(old_value, new_value, criterion=0.001)
            pbar.update()
            pbar.set_description('iteration {}, max difference = {}'.format(times, max_diff))
            if tag:
                break

    except KeyboardInterrupt:
        return new_value

    finally:
        pbar.close()

    return new_value






if __name__ == '__main__':
    input_shape = [32, 32, 3]
    num_class = 10
    batch_size = 50
    regu_weight = 0.01
    learning_rate = 0.0005

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, num_class)
    y_test = to_categorical(y_test, num_class)

    constructor = Cifar10_Classifier(input_shape, num_class, regu_weight)

    population = 100
    data = np.split(x_train, population)
    label = np.split(y_train, population)

    valuation = data_valuation(data, label, constructor)
    np.save('data/valuation.npy', valuation)



        


