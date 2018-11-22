from keras import backend as K
from keras.engine.topology import Layer


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

    def __init__(self, filter_shape, num_layers, **kwargs):
        self.filter_shape= filter_shape
        self.num_layers= num_layers
        super(MyLayerRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        y= int(input_shape[3]/2)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.filter_shape,
                                        self.filter_shape, y,self.num_layers),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayerRelu, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        y=int(K.int_shape(x)[3]/2)
        return K.dot()
        return K.concatenate([K.conv2d(x[:,:,:,0:y],self.kernel),
          K.conv2d(x[:,:,:,y:2*y],self.kernel)])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]-self.filter_shape+1,
          input_shape[1]-self.filter_shape+1,2*self.num_layers)


class MyLayerDropout(Layer):

	def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        # self.supports_masking = True
        super(MyLayerDropout, self).__init__(**kwargs)

    # def _get_noise_shape(self, inputs):
    #     if self.noise_shape is None:
    #         return self.noise_shape

    #     symbolic_shape = K.shape(inputs)
    #     noise_shape = [symbolic_shape[axis] if shape is None else shape
    #                    for axis, shape in enumerate(self.noise_shape)]
    #     return tuple(noise_shape)

    def call(self, x):
        if 0. < self.rate < 1.:

            def dropped_inputs():
            	y=int(K.int_shape(x)[3]/2)
                return K.concatenate([K.dropout(x[:,:,:,0:y], self.rate,self.noise_shape,seed=self.seed),
                	K.dropout(x[:,:,:,y:K.int_shape(x)[3]],self.rate,self.noise_shape,seed=self.seed)])
            return K.in_train_phase(dropped_inputs, x,
                                    training=training)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class MyLayerMaxPooling(Layer):

