from keras.layers import Dense
from keras.layers import BatchNormalization, Activation, Dropout
from keras.models import Sequential
from keras.datasets import mnist
from keras.objectives import binary_crossentropy
from keras import backend as K
import tensorflow as tf
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

with open("AutoEncoder.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def load_mnist():
    print("mnist dataset is loaded...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print("x_train.shape = " + str(x_train.shape))
    print("y_train.shape = " + str(y_train.shape))
    print("x_test.shape = " + str(x_test.shape))
    print("y_test.shape = " + str(y_test.shape))
    return x_train, y_train, x_test, y_test


class AutoEncoder(object):
    def __init__(self, ae_shape, learning_rate, batch_size, epochs, num_decay_steps, decay, test_log_every, test_size):
        self.ae_shape = ae_shape
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_decay_steps = num_decay_steps
        self.decay = decay
        self.test_log_every = test_log_every
        self.test_size = test_size

        self.model = self.build_model()
        self.create_model()

    def print_layer(self,prev,now):
        print("%5d  -->  %5d"%(self.ae_shape[prev], self.ae_shape[now]))

    def build_model(self):
        print("building model...")
        sh=self.ae_shape
        model = Sequential() # ae_shape=[784,500,200]
        model.add(Dense(sh[1], input_shape=(sh[0],)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        self.print_layer(0,1)
        if len(sh) >= 3:
            for i in range(2,len(sh)):
                model.add(Dense(sh[i]))
                # model.add(BatchNormalization()) # learning과 test때 달라서 keras_learning_phase를 feed_dict에 추가해야함
                model.add(Activation('relu'))
                self.print_layer(i-1,i)
            for i in range(len(sh)-2,0,-1):
                model.add(Dense(sh[i]))
                # model.add(BatchNormalization())
                model.add(Activation('relu'))
                self.print_layer(i+1,i)
        model.add(Dense(sh[0]))
        # model.add(BatchNormalization())
        model.add(Activation('sigmoid'))
        self.print_layer(1,0)
        print(model.summary())
        return model

    def optimizer(self, loss, init_learning_rate, name):
        with tf.variable_scope(name) as scope:
            #step = tf.Variable(0, name='step')  # 현재 iteration을 표시해 줄 변수
            #learning_rate = tf.train.exponential_decay(init_learning_rate, step, self.num_decay_steps, self.decay, staircase=True)  # learning_rate = (initial_learning_rate)*decay^(int(step/num_decay_step))
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=step)  # var_list의 변수들을 update
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='Adam').minimize(loss)
        return optimizer

    def create_model(self):
        with tf.variable_scope("input") as scope:
            self.input = tf.placeholder(tf.float32, shape=(None, self.ae_shape[0]))
        with tf.variable_scope("loss") as scope:
            self.train_loss = tf.reduce_mean(tf.pow(self.input - self.model(self.input), 2), name='train_loss')
            self.test_loss = tf.reduce_mean(tf.pow(self.input - self.model(self.input), 2), name='test_loss')
            self.train_loss_summary = tf.summary.scalar('train_loss', self.train_loss)
            self.test_loss_summary = tf.summary.scalar('test_loss', self.test_loss)
        with tf.variable_scope("optimizer") as scope:
            self.opt = self.optimizer(self.train_loss, self.learning_rate, name='optimizer')

    def train_model(self):
        # config = tf.ConfigProto(log_device_placement=True)
        # config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession()#config=config)
        with K.tf.device('/gpu:0'):
            K.set_session(sess)

        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter("C:/tmp/AutoEncoder", sess.graph)

        x_train, _, x_test, _ = load_mnist()
        total_batch = int(x_train.shape[0] // self.batch_size)
        for epoch in range(self.epochs):
            np.random.shuffle(x_train)
            for step in range(total_batch):
                train_loss, _ = sess.run([self.train_loss_summary, self.opt], feed_dict={self.input: x_train[step*self.batch_size:(step+1)*self.batch_size], K.learning_phase(): 1})
                writer.add_summary(train_loss, (epoch * total_batch + step))
                if step % self.test_log_every == 0:
                    np.random.shuffle(x_test)
                    test_loss = sess.run(self.test_loss_summary, feed_dict={self.input:x_test[0:self.test_size], K.learning_phase(): 0})
                    writer.add_summary(test_loss, (epoch * total_batch + step))
                    print('\repoch %d, step %d is done!, train_loss: %4f, test_loss: %4f' % (epoch + 1, step + 1,
                        self.train_loss.eval(feed_dict={self.input: x_train[step * self.batch_size:(step + 1) * self.batch_size], K.learning_phase(): 1}),
                        self.test_loss.eval(feed_dict={self.input:x_test[0:self.test_size], K.learning_phase(): 0})), end=' ')
        tf.summary.FileWriter.close(writer)
        print('\n=========== training end ===========')

        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(sess.run(self.model(self.input),{self.input:x_test[i].reshape(1,784), K.learning_phase(): 0}).reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        sess.close()


    """
        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        """

ae= AutoEncoder([784,500,400,300,200,128,100], 0.01, 100, 500, 1000, 0.95, 10, 1000) # ae_shape, learning_rate, batch_size, epochs, num_decay_steps, decay, test_log_every, test_size
ae.train_model()
