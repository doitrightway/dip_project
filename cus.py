from keras import backend as K
from keras.engine.topology import Layer
import tensorflow_probability as tfp

class MyLayer(Layer):

    def __init__(self, filter_shape, num_layers, **kwargs):
        self.filter_shape= filter_shape
        self.num_layers= num_layers
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        y= int(input_shape[3]/2)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.filter_shape,
                                        self.filter_shape, y,self.num_layers),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        y=int(K.int_shape(x)[3]/2)
        return K.concatenate([K.conv2d(x[:,:,:,0:y],self.kernel),
          K.conv2d(x[:,:,:,y:2*y],self.kernel)])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]-self.filter_shape+1,
          input_shape[1]-self.filter_shape+1,2*self.num_layers)


class MyLayerDense(Layer):

  def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayerDense, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        y=int(input_shape[1]/2)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(y, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayerDense, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        y=int(K.int_shape(x)[3]/2)
        return K.concatenate([K.dot(x[:,0:y], self.kernel),K.dot(x[:,y:2*y], self.kernel)])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class MyLayerRelu(Layer):

    def __init__(self, **kwargs):
        super(MyLayerRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(MyLayerRelu, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        y=int(K.int_shape(x)[3]/2)
        z= Lambda(lambda inputs: inputs[0]/inputs[1] if inputs[1]!=0 else 1000000)([x[:,:,:,0:y],x[:,:,:,y:2*y]])
        tfd = tfp.distributions
        dist = tfd.Normal(loc=0., scale=1.)
        var1 = dist.cdf(z)
        var2 = dist.prob(z)
        mean = K.dot(x[:,:,:,0:y],var1) + K.dot(x[:,:,:,y:2*y],var2)
        var3 = K.dot(x[:,:,:,0:y]+x[:,:,:,y:2*y],var1) 
        var4 = K.dot(K.dot(x[:,:,:,0:y],x[:,:,:,y:2*y]),var2)
        variance = var3+var4-K.square(mean)
        return K.concatenate([mean,variance])

    def compute_output_shape(self, input_shape):
        return input_shape


class MyFlatten(Layer):

    def __init__(self, **kwargs):
        super(MyLayerRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(MyLayerRelu, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        y=int(K.int_shape(x)[3]/2)
        return K.concatenate([K.batch_flatten(x[:,:,:,0:y]), K.batch_flatten(x[:,:,:,y:2*y])])

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1]*input_shape[2]*input_shape[3])




