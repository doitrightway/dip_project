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


class MyLayerDenseRelu(Layer):

    def __init__(self, **kwargs):
        super(MyLayerDenseRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(MyLayerDenseRelu, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        y=int(K.int_shape(x)[1]/2)
        z= Lambda(lambda inputs: inputs[0]/inputs[1] if inputs[1]!=0 else 1000000)([x[:,0:y],x[:,y:2*y]])
        tfd = tfp.distributions
        dist = tfd.Normal(loc=0., scale=1.)
        var1 = dist.cdf(z)
        var2 = dist.prob(z)
        mean = K.dot(x[:,0:y],var1) + K.dot(x[:,y:2*y],var2)
        var3 = K.dot(x[:,0:y]+x[:,y:2*y],var1) 
        var4 = K.dot(K.dot(x[:,0:y],x[:,y:2*y]),var2)
        variance = var3+var4-K.square(mean)
        return K.concatenate([mean,variance])

    def compute_output_shape(self, input_shape):
        return input_shape


class MyFlatten(Layer):

    def __init__(self, **kwargs):
        super(MyFlatten, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(MyFlatten, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        y=int(K.int_shape(x)[3]/2)
        return K.concatenate([K.batch_flatten(x[:,:,:,0:y]), K.batch_flatten(x[:,:,:,y:2*y])])

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1]*input_shape[2]*input_shape[3])





class MyLayerDropout(Layer):

	def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        super(MyLayerDropout, self).__init__(**kwargs)

  def call(self, x):
        if 0. < self.rate < 1.:

            def dropped_inputs():
            	y=int(K.int_shape(x)[3]/2)
                return K.concatenate([K.dropout(x[:,:,:,0:y], self.rate, self.noise_shape, seed=self.seed),
                	K.dropout(x[:,:,:,y:K.int_shape(x)[3]],self.rate, self.noise_shape, seed = self.seed)])
            return K.in_train_phase(dropped_inputs, x,
                                    training=training)
        return x

  def compute_output_shape(self, input_shape):
      return input_shape


class MyLayerDenseDropout(Layer):

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        super(MyLayerDenseDropout, self).__init__(**kwargs)

  def call(self, x):
        if 0. < self.rate < 1.:

            def dropped_inputs():
              y=int(K.int_shape(x)[3]/2)
                return K.concatenate([K.dropout(x[:,0:y], self.rate, self.noise_shape, seed=self.seed),
                  K.dropout(x[:,y:K.int_shape(x)[1]],self.rate, self.noise_shape, seed = self.seed)])
            return K.in_train_phase(dropped_inputs, x,
                                    training=training)
        return x

  def compute_output_shape(self, input_shape):
      return input_shape


class DirichletLayer(Layer):

  def __init__(self, c1,c2,**kwargs):
        self.c1=c1
        self.c2=c2
        super(DirichletLayer, self).__init__(**kwargs)

  def call(self, x):
      y=int(K.int_shape(x)[1]/2)
      var = K.softmax(x[:,0:y])
      scale = self.c1 + self.c2*K.sqrt(K.sum(K.dot(var,x[:,y,2*y]),axis=-1))
      scale = K.expand_dims(scale,axis=-1)
      scale = K.repeat_elements(scale,y,axis=-1)
      return var/scale

  def compute_output_shape(self, input_shape):
      y=int(input_shape[1]/2)
      return (input_shape[0],y)

