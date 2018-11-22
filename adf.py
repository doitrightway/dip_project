from keras.models import Sequential
from keras.layers import Dense
from layers import MyLayer
from layers import MyLayerDense
from layers import MyLayerRelu
from layers import MyLayerDenseRelu
from layers import MyLayerDenseDropout
from layers import MyLayerDropout
from layers import MyFlatten
from layers import DirichletLayer
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import keras.backend as K
import numpy as np
import keras
# import matplotlib.pyplot as plt
import tensorflow as tf


def customloss(y_true, y_pred):
	y_true = K.clip(y_true, K.epsilon(), 1)
	opt1 = K.sum(K.dot((y_pred-1),K.log(y_true)),axis=-1)
	opt2 = K.sum(K.exp(tf.lgamma(y_pred)),axis=-1)
	opt3 = K.exp(tf.lgamma(K.sum(y_pred,axis=-1)))
	return opt2-opt1-opt3

num_classes=10
batch_size = 128
epochs=10
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# for i in range(9):
#   plt.subplot(3,3,i+1)
#   plt.tight_layout()
#   plt.imshow(x_train[i], cmap='gray', interpolation='none')
#   plt.title("Digit: {}".format(y_train[i]))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()

x_train=x_train/255;
y_train=y_train/255;

x_train = x_train.reshape((np.shape(x_train)+(1,)))
x_test = x_test.reshape((np.shape(x_test)+(1,)))

train_shape = np.shape(x_train)

noise_train = 0.01*np.random.randn(train_shape[0],train_shape[1],train_shape[2],train_shape[3])
x_train= np.concatenate([x_train,noise_train],axis=-1)

test_shape = np.shape(x_test)
noise_test = 0.01*np.random.randn(test_shape[0],test_shape[1],test_shape[2],test_shape[3])
x_test= np.concatenate([x_test,noise_test])



print(np.shape(x_train))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


y_train = y_train + 0.001*K.random_normal(K.shape(y_train))
y_test= y_test + 0.001*K.random_normal(K.shape(y_test))

model = Sequential()

model.add(MyLayer(filter_shape=3,num_layers=32))
model.add(MyLayerRelu())
model.add(MyLayer(filter_shape= 3,num_layers=64))
model.add(MyLayerRelu())
model.add(MyLayerDropout(0.25, seed =0))
model.add(MyFlatten())
model.add(MyLayerDense(128))
model.add(MyLayerDenseRelu())
model.add(MyLayerDenseDropout(0.5, seed=0))
model.add(MyLayerDense(10))
model.add(DirichletLayer(0.5,0.5))


sgd = SGD(lr=0.01, decay=0.003, momentum=0.9, nesterov=True)
model.compile(loss=customloss, optimizer=sgd, metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

