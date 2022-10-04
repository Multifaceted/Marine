import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def posterior_mean_field_with_initializer(kernel_size, bias_size=0, dtype=None, initializer=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype, initializer=initializer),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1))
  ])

def prior_trainable_with_initializer(kernel_size, bias_size=0, dtype=None, initializer=None):
    print("kernel size:", kernel_size)
    print("bias size:", bias_size)
    n = kernel_size + bias_size
    return tf.keras.Sequential([
          tfp.layers.VariableLayer(n, dtype=dtype, initializer=initializer),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
              tfd.Normal(loc=t, scale=1),
              reinterpreted_batch_ndims=1)),
  ])


# def print_param(t):
#   # print(t)
#   return tfd.Independent(
#           tfd.Normal(loc=t[..., :],
#                      scale=1e-5 + tf.nn.softplus( np.log(np.expm1(1.)) + t[..., 4:])), reinterpreted_batch_ndims=1)
# prior_trainable_det = partial(prior_trainable, initializer=)