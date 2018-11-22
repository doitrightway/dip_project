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
import matplotlib.pyplot as plt
import tensorflow as tf


def customloss(y_true, y_pred):
	c1=0.5
	c2=0.5
	y=int(K.int_shape(y_pred)[1]/2)
	var = K.softmax(y_pred[:,0:y])
	scale = c1 + c2*K.sqrt(K.sum(var*y_pred[:,y:2*y],axis=-1))
	scale = K.expand_dims(scale,axis=-1)
	scale = K.repeat_elements(scale,y,axis=-1)
	output = var/scale
	y_true = K.clip(y_true, K.epsilon(), 1)
	opt1 = K.sum((output-1)*K.log(y_true),axis=-1)
	opt2 = K.sum(K.exp(tf.lgamma(output)),axis=-1)
	opt3 = K.exp(tf.lgamma(K.sum(output,axis=-1)))
	return opt2-opt1-opt3

	


def custom_metric(y_true, y_pred):
	y=int(K.int_shape(y_pred)[1]/2)
	return keras.metrics.categorical_accuracy(y_true, y_pred[:,0:y])



num_classes=10
batch_size = 128
epochs=10
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
x_test= x_test/255;

x_train = x_train.reshape((np.shape(x_train)+(1,)))
x_test = x_test.reshape((np.shape(x_test)+(1,)))



print(np.shape(x_train))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


y_train = y_train + 0.001*np.random.randn(np.shape(y_train)[0],np.shape(y_train)[1])
y_test= y_test + 0.001*np.random.randn(np.shape(y_test)[0],np.shape(y_test)[1])


model = Sequential()



model.add(MyLayer(filter_shape=3,num_layers=5))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
#32 convolution filters used each of size 3x3
#again
model.add(Conv2D(64, (3, 3), activation='relu'))
#64 convolution filters used each of size 3x3
#choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))	
#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))
#flatten since too many dimensions, we only want a classification output
model.add(Flatten())
#fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
#one more dropout for convergence' sake :) 
model.add(Dropout(0.5))
#output a softmax to squash the matrix into output probabilities
model.add(Dense(20))


sgd = SGD(lr=0.01, decay=0.003, momentum=0.9, nesterov=True)
model.compile(loss=customloss, optimizer=sgd, metrics=[custom_metric])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

