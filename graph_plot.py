from keras.models import load_model,model_from_json
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
	opt2 = K.sum(tf.lgamma(output),axis=-1)
	opt3 = tf.lgamma(K.sum(output,axis=-1))
	return opt2-opt1-opt3

def custom_metric(y_true, y_pred):
	y=int(K.int_shape(y_pred)[1]/2)
	return keras.metrics.categorical_accuracy(y_true, y_pred[:,0:y])


keras.losses.customloss = customloss
keras.metrics.custom_metric = custom_metric
model = load_model('probout_network.05.hdf5')
sgd = SGD(lr=0.01, decay=.003, momentum=0.9, nesterov=True)

model_json = model.to_json()
with open("probout_network.05.json","w") as json_file:
	json_file.write(model_json)

model.save_weights("probout_network.05.h5")
print("Saved model to disk")


json_file= open("probout_network.05.json",'r')
loaded_model_json= json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("probout_network.05.h5")
print("Loaded model from disk")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_size=128*64*4
test_size = 128*64
x_train = x_train[0:train_size,:,:]
y_train = y_train[0:train_size]
x_test = x_test[0:test_size,:,:]
y_test = y_test[0:test_size]

x_train = x_train.reshape((np.shape(x_train)+(1,)))
x_test = x_test.reshape((np.shape(x_test)+(1,)))

inp = loaded_model.input                                           # input placeholder
outputs = [layer.output for layer in loaded_model.layers]          # all layer outputs
functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions
# Testing
test = x_test
layer_outs = [func([test]) for func in functors]
# print(layer_outs[-1][0])
entropy_x = [0 for i in range(test_size)]
entropy_y = [0 for i in range(test_size)]

for i in range(test_size):
	vec = layer_outs[-1][0][i]
	mean = vec[0:10]
	variance = vec[10:20]
	idx = np.argmax(mean)
	true_idx = y_test[i]
	mean = np.exp(mean)/np.sum(np.exp(mean))
	variance = np.exp(variance)/np.sum(np.exp(variance))
	entropy_x[i] = -np.sum(mean*np.log(mean))/10
	entropy_y[i] = -np.log(variance[true_idx])*variance[idx]

plt.plot(entropy_x,entropy_y)
plt.show()







