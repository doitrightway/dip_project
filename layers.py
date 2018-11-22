from keras import backend as K
from keras.engine.topology import Layer
import tensorflow_probability as tfp
import tensorflow as tf
from keras.layers import Lambda
from keras.layers.pooling import _Pooling2D
import keras

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
                                      initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        y=int(K.int_shape(x)[3]/2)
        return K.concatenate([K.conv2d(x[:,:,:,0:y],self.kernel),
          K.conv2d(x[:,:,:,y:2*y],K.square(self.kernel))])

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
                                        initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                                        trainable=True)
          super(MyLayerDense, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        y=int(K.int_shape(x)[1]/2)
        return K.concatenate([K.dot(x[:,0:y],self.kernel),K.dot(x[:,y:2*y], K.square(self.kernel))])

    def compute_output_shape(self, input_shape):
          return (input_shape[0], 2*self.output_dim)


class MyLayerRelu(Layer):

    def __init__(self, **kwargs):
        super(MyLayerRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(MyLayerRelu, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        y=int(K.int_shape(x)[3]/2)
        z=x[:,:,:,0:y]/K.sqrt(x[:,:,:,y:2*y])
        tfd = tfp.distributions
        dist = tfd.Normal(loc=0., scale=1.)
        var1 = dist.cdf(z)
        var2 = dist.prob(z)
        mean = x[:,:,:,0:y]*var1 + K.sqrt(x[:,:,:,y:2*y])*var2
        var3 = (K.square(x[:,:,:,0:y])+x[:,:,:,y:2*y])*var1
        var4 = x[:,:,:,0:y]*K.sqrt(x[:,:,:,y:2*y])*var2
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
        z=x[:,0:y]/K.sqrt(x[:,y:2*y])
        tfd = tfp.distributions
        dist = tfd.Normal(loc=0., scale=1.)
        var1 = dist.cdf(z)
        var2 = dist.prob(z)
        mean = x[:,0:y]*var1 + K.sqrt(x[:,y:2*y])*var2
        var3 = (K.square(x[:,0:y])+x[:,y:2*y])*var1 
        var4 = x[:,0:y]*K.sqrt(x[:,y:2*y])*var2
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

################

def dropped_inputs(x, rate, noise_shape, seed):
    y=int(K.int_shape(x)[3]/2)
    return K.concatenate([K.dropout(x[:,:,:,0:y], rate, noise_shape, seed=seed),
      K.dropout(x[:,:,:,y:K.int_shape(x)[3]],rate, noise_shape, seed = seed)])


def dropped_dense_inputs(x, rate, noise_shape, seed):
    y=int(K.int_shape(x)[1]/2)
    return K.concatenate([K.dropout(x[:,0:y], rate, noise_shape, seed=seed),
      K.dropout(x[:,y:K.int_shape(x)[1]],rate, noise_shape, seed = seed)])

##############

class MyLayerDropout(Layer):

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        super(MyLayerDropout, self).__init__(**kwargs)

    def call(self, x, training=None):
        if 0. < self.rate < 1.:
            return K.in_train_phase((lambda : dropped_inputs(x, self.rate, self.noise_shape, self.seed)), x,
                                    training=training)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape



#############

class MyLayerDenseDropout(Layer):

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        super(MyLayerDenseDropout, self).__init__(**kwargs)

    def call(self, x, training= None):
        if 0. < self.rate < 1.:
            return K.in_train_phase((lambda : dropped_dense_inputs(x, self.rate, self.noise_shape, self.seed)), x,
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
        scale = self.c1 + self.c2*K.sqrt(K.sum(var*x[:,y:2*y],axis=-1))
        scale = K.expand_dims(scale,axis=-1)
        scale = K.repeat_elements(scale,y,axis=-1)
        return var/scale

    def compute_output_shape(self, input_shape):
        y=int(input_shape[1]/2)
        return (input_shape[0],y)


class MyLayerMaxPool(_Pooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(MyLayerMaxPool, self).__init__(pool_size, strides, padding,data_format, **kwargs)


    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        y=K.int_shape(x)
        i1=range(0,y[1],2)
        i2=range(1,y[1],2)
        j1=range(0,y[2],2)
        j2=range(1,y[2],2)
        p1=[[[[k,i,j] for j in j1] for i in i1] for k in range(0,y[0])]
        p2=[[[[k,i,j] for j in j2] for i in i1] for k in range(0,y[0])]
        p3=[[[[k,i,j] for j in j1] for i in i2] for k in range(0,y[0])]
        p4=[[[[k,i,j] for j in j2] for i in i2] for k in range(0,y[0])]
        x1=tf.gather(inputs,p1)
        x2=tf.gather(inputs,p2)
        x3=tf.gather(inputs,p3)
        x4=tf.gather(inputs,p4)
        def get_mean(a1,a2):
            m1=a1[:,:,:,0:y[3]/2]
            m2=a2[:,:,:,0:y[3]/2]
            v1=a1[:,:,:,y[3]/2:y[3]]
            v2=a2[:,:,:,y[3]/2:y[3]]
            di=m1-m2
            sum_sq=K.sqrt(v1*v1+v2*v2)
            alpha=di/sum_sq
            tfd = tfp.distributions
            dist = tfd.Normal(loc=0., scale=1.)
            var1 = dist.cdf(alpha)
            var2 = dist.prob(alpha)
            mean1=sum_sq*var2+di*var1+m2
            variance1=(m1+m2)*sum_sq*var2+(m1*m1+v1*v1)*var1+(m2*m2+v2*v2)*(1-var1)-mean1*mean1
            return K.concatenate([mean1,variance1])

        return get_mean(get_mean(x1,x3),get_mean(x2,x4))



