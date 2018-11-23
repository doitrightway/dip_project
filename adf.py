from keras.models import Sequential
from keras.layers import Dense
from layers import MyLayer
from layers import MyLayerDense
from layers import MyLayerRelu
from layers import MyLayerDenseRelu
from layers import MyLayerDenseDropout
from layers import MyLayerDropout
from layers import MyFlatten
from layers import MyLayerMaxPool
from layers import DirichletLayer
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import keras.backend as K
from keras import metrics
import numpy as np
import keras
# import matplotlib.pyplot as plt
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
	opt2 = K.sum(tf.lgamma(output),axis=-1)
	opt3 = tf.lgamma(K.sum(output,axis=-1))
	return opt2-opt1-opt3


def gaussianloss(y_true, y_pred):
	y=int(K.int_shape(y_pred)[1]/2)
	mean= y_pred[:,0:y]
	variance= y_pred[:,y:2*y]
	variance = K.clip(variance, K.epsilon(), 1)
	return K.sum(K.square(y_true-mean)/variance + K.log(variance),axis=-1)

def meanloss(y_true, y_pred):
	y=int(K.int_shape(y_pred)[1]/2)
	mean= y_pred[:,0:y]
	return K.sum(K.abs(y_true-mean),axis=-1)


def custom_metric(y_true, y_pred):
	y=int(K.int_shape(y_pred)[1]/2)
	return keras.metrics.categorical_accuracy(y_true, y_pred[:,0:y])


num_classes=10
batch_size = 128
epochs=1
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()


train_size=128*64*4
test_size = 128*64
x_train = x_train[0:train_size,:,:]
y_train = y_train[0:train_size]
x_test = x_test[0:test_size,:,:]
y_test = y_test[0:test_size]

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

train_shape = np.shape(x_train)
noise_train = 0.01*np.abs(np.random.randn(train_shape[0],train_shape[1],train_shape[2],train_shape[3]))
x_train= np.concatenate([x_train,noise_train],axis=-1)

test_shape = np.shape(x_test)
noise_test = 0.01*np.abs(np.random.randn(test_shape[0],test_shape[1],test_shape[2],test_shape[3]))
x_test= np.concatenate([x_test,noise_test],axis=-1)


# print(np.shape(x_train))



y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


y_train = y_train + 0.001*np.random.randn(np.shape(y_train)[0],np.shape(y_train)[1])
y_test= y_test + 0.001*np.random.randn(np.shape(y_test)[0],np.shape(y_test)[1])

model = Sequential()

model.add(MyLayer(filter_shape=3,num_layers=32))
model.add(MyLayerRelu())
model.add(MyLayer(filter_shape= 3,num_layers=64))
model.add(MyLayerMaxPool(pool_size=(2, 2)))
model.add(MyLayerRelu())
model.add(MyLayerDropout(0.25, seed =0))
model.add(MyFlatten())
model.add(MyLayerDense(128))
model.add(MyLayerDenseRelu())
model.add(MyLayerDenseDropout(0.5, seed=0))
model.add(MyLayerDense(10))

checkpoint = keras.callbacks.ModelCheckpoint('adfnetwork.{epoch:02d}.hdf5', 
	monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

sgd = SGD(lr=0.01, decay=0.003, momentum=0.9, nesterov=True)
model.compile(loss=customloss, optimizer=sgd, metrics=[custom_metric])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[checkpoint]
          validation_data=(x_test, y_test))

# im= keras.models.Model(inputs = model.input, outputs=model.layers[0].output)
# imd= im.predict(x_train)
# print(imd)

