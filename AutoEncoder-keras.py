from keras.layers import Dense
from keras.layers import BatchNormalization, Activation, Dropout
from keras.models import Sequential
from keras.datasets import mnist
from keras.objectives import binary_crossentropy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
import tensorflow as tf
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
from keras.optimizers import RMSprop, Adam
import csv

with open("AutoEncoder.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_data(data_dir, axis, threshold):
    """
    load raw data
    :return: data matrix with None
    """
    # load csv file
    with open(data_dir, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = np.asarray(data, dtype=float)
    with open('./data/PHR/example_sparsity.csv', 'r') as f:
        reader = csv.reader(f)
        example_sparsity = list(reader)
        example_sparsity = np.asarray(example_sparsity, dtype=float)
    with open('./data/PHR/feature_sparsity.csv', 'r') as f:
        reader = csv.reader(f)
        feature_sparsity = list(reader)
        feature_sparsity = np.asarray(feature_sparsity, dtype=float)
    print('csv file is loaded')

    if axis == 0: # row
        x_train=[]
        x_test=[]
        train_idx = []
        test_idx = []
        for i in range(data.shape[0]):
            if example_sparsity[i] < threshold:
                x_train.append(data[i])
                train_idx.append(i+1)
            else:
                x_test.append(data[i])
                test_idx.append(i+1)
        #x_test=x_train[1000:1214]
        #x_train=x_train[0:1000]
        y_train=x_train
        y_test = x_test
    elif axis ==  1: # column
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        train_idx = []
        test_idx = []
        for i in range(data.shape[0]):
            tmp = data[i]
            if example_sparsity[i] < threshold:
                x_train.append(tmp[57:115])
                y_train.append(tmp[0:57])
            else:
                x_test.append(tmp[57:115])
                y_test.append(tmp[0:57])
    else:
        raise(ValueError)

    return data, np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), np.asarray(train_idx), np.asarray(test_idx)


class AutoEncoder(object):
    def __init__(self, ae_shape, axis, learning_rate, batch_size, epochs, num_decay_steps, decay, test_log_every, test_size):
        self.ae_shape = ae_shape
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_decay_steps = num_decay_steps
        self.decay = decay
        self.test_log_every = test_log_every
        self.test_size = test_size
        self.axis = axis
        self.model = self.build_model()
        #self.create_model()

    def print_layer(self,prev,now):
        print("%5d  -->  %5d"%(self.ae_shape[prev], self.ae_shape[now]))

    def build_model(self):
        print("building model...")
        sh=self.ae_shape
        model = Sequential() # ae_shape=[784,500,200]
        model.add(Dense(sh[1], input_shape=(sh[0],)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(Dropout(0.25))
        self.print_layer(0,1)
        if len(sh) >= 3:
            for i in range(2,len(sh)):
                model.add(Dense(sh[i]))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                #model.add(Dropout(0.25))
                self.print_layer(i-1,i)
            for i in range(len(sh)-2,0,-1):
                model.add(Dense(sh[i]))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                #model.add(Dropout(0.25))
                self.print_layer(i+1,i)
        if self.axis == 0:
            model.add(Dense(sh[0])) # example
        elif self.axis == 1:
            model.add(Dense(57)) # feature
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))
        self.print_layer(1,0)
        print(model.summary())
        return model

    def train_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.log_device_placement = False
        config.allow_soft_placement = True
        sess = tf.InteractiveSession(config=config)
        with K.tf.device('/gpu:0'):
            K.set_session(sess)

        data, x_train, y_train, x_test, y_test, train_idx, test_idx = load_data('./data/PHR/zero_to_one.csv', axis=self.axis, threshold=0.3)
        total_batch = int(x_train.shape[0] // self.batch_size)

        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate))
        history = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, shuffle=True, validation_data=(x_test,y_test), verbose=2)

        # print(x_test[0:2])
        # output = self.model.predict(x_test[0:2])
        # print(output)

        result = self.model.predict(data)

        with open('row_padding_ae_full.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in range(len(result)):
                writer.writerow(result[i])
            print('result saved!')

        '''
        result = self.model.predict(x_test)

        with open('row_padding_ae.csv','a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in range(len(result)):
                tmp = np.concatenate((np.asarray([test_idx[i]]),result[i]))
                writer.writerow(tmp)
            print('result saved!')
        '''
# [115,80] # row, [58.30] # col

if __name__ == "__main__":
    ae= AutoEncoder(config['ae_shape'], config['axis'], config['init_learning_rate'], config['batch_size'], config['epochs'],
                    config['num_decay_steps'], config['decay'], config['log_every'], config['test_size'])
    # ae_shape, learning_rate, batch_size, epochs, num_decay_steps, decay, test_log_every, test_size
    ae.train_model()