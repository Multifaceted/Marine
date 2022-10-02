from util.load_data import read_pipeline
from util.interpolate import interpolate
from util.model import init_model_stochastic
from util.prior_posterior import  posterior_mean_field_with_initializer, prior_trainable_with_initializer
from util.plot import plot_average
from functools import partial
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf
import numpy as np



posterior_mean_field = partial(posterior_mean_field_with_initializer, initializer=tf.keras.initializers.Constant([5., 6, 7, 8, 0, 0, 0, 0]))
prior_trainable = partial(prior_trainable_with_initializer, initializer=tf.keras.initializers.Constant([0., 1, 2, 3]))

model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(1, )),
    
  tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1/2),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
])
n_mc = 1000

res_1 = [model(tf.constant([[1]], dtype=tf.float32)) for _ in range(n_mc)]
res_1_loc = np.mean([_.mean() for _ in res_1]) # 12 = 5 * 1 + 7
res_1_scale = np.mean([_.stddev() for _ in res_1]) # 0.76631665 = 1e-3 + tf.math.softplus(0.01 * (6 + 8))
res_1_loc_std = np.std([_.mean() for _ in res_1]) # 1.3837872 = np.sqrt(1 + 1)
res_1_scale_std =  np.std([_.stddev() for _ in res_1]) 
# 0.0071263276 = np.std([1e-3 + tf.math.softplus(0.01 * np.random.normal(loc=14, scale=np.sqrt(2))) for _ in range(1000)])

# Verify
res_2 = [model(tf.constant([[2]], dtype=tf.float32)) for _ in range(n_mc)]
res_2_loc = np.mean([_.mean() for _ in res_2]) # 17 = 5 * 2 + 7
res_2_scale = np.mean([_.stddev() for _ in res_2]) # 0.79953605 = 1e-3 +  tf.math.softplus(0.01 * (6 * 2 + 8))
res_2_loc_std = np.std([_.mean() for _ in res_2])  # 2.2641742 = np.sqrt(1*2**2 + 1)


###############################################

posterior_mean_field = partial(posterior_mean_field_with_initializer, initializer=tf.keras.initializers.Constant([0, 0, 0, 0, 5., 6, 7, 8]))
prior_trainable = partial(prior_trainable_with_initializer, initializer=tf.keras.initializers.Constant([0., 1, 2, 3]))

model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(1, )),
    
  tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1/2),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
])
n_mc = 1000

res_1 = [model(tf.constant([[1]], dtype=tf.float32)) for _ in range(n_mc)]
res_1_loc = np.mean([_.mean() for _ in res_1]) # 0 = 5 * 0 + 7 * 0
res_1_scale = np.mean([_.stddev() for _ in res_1]) # 0.695651 = 1e-3 + tf.math.softplus(0.01 * 0)
res_1_loc_std = np.std([_.mean() for _ in res_1]) # 8.946621 = np.sqrt(5**2 + 7**2)
res_1_scale_std =  np.std([_.stddev() for _ in res_1]) 
# 0.0538057 = np.std([1e-3 + tf.math.softplus(0.01 * np.random.normal(loc=0, scale=np.sqrt(6**2 + 8 ** 2))) for _ in range(1000)])

res_2 = [model(tf.constant([[2]], dtype=tf.float32)) for _ in range(n_mc)]
res_2_loc = np.mean([_.mean() for _ in res_2]) # 0 = 5 * 0 + 7 * 0
res_2_scale = np.mean([_.stddev() for _ in res_2]) # 0.695651 = 1e-3 + tf.math.softplus(0.01 * 0)
res_2_loc_std = np.std([_.mean() for _ in res_2]) # 8.946621 = np.sqrt((5 * 2)**2 + 7**2)
res_2_scale_std =  np.std([_.stddev() for _ in res_2]) 
# 0.0771139 = np.std([1e-3 + tf.math.softplus(0.01 * np.random.normal(loc=0, scale=np.sqrt(12**2 + 8 ** 2))) for _ in range(1000)])

#################################################
"""
Neuron 1 for loc mean
Neuron 1 for stddev mean
Neuron 2 for loc mean
Neuron 2 for stddev mean
Bias for loc mean
Bias for stddev mean

Neuron 1 for loc scale
Neuron 1 for stddev scale
Neuron 2 for loc scale
Neuron 2 for stddev scale
Bias for loc scale
Bias for stddev scale
"""
posterior_mean_field = partial(posterior_mean_field_with_initializer, initializer=tf.keras.initializers.Constant([0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0]))
prior_trainable = partial(prior_trainable_with_initializer, initializer="zeros")

model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(2, )),
    
  tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1/2),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
])
n_mc = 1000

res_1 = [model(tf.constant([[0, 1]], dtype=tf.float32)) for _ in range(n_mc)]
res_1_loc = np.mean([_.mean() for _ in res_1]) # 12 = 5 * 1 + 7
res_1_scale = np.mean([_.stddev() for _ in res_1]) # 0.76631665 = 1e-3 + tf.math.softplus(0.01 * (6 + 8))
res_1_loc_std = np.std([_.mean() for _ in res_1]) # 1.3837872 = np.sqrt(1 + 1)
res_1_scale_std =  np.std([_.stddev() for _ in res_1]) 
# 0.0071263276 = np.std([1e-3 + tf.math.softplus(0.01 * np.random.normal(loc=14, scale=np.sqrt(2))) for _ in range(1000)])

##################################################

posterior_mean_field = partial(posterior_mean_field_with_initializer, initializer=tf.keras.initializers.Constant([5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0]))
prior_trainable = partial(prior_trainable_with_initializer, initializer="zeros")

model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(2, )),
    
  tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1/2),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
])
n_mc = 1000

res_1_loc = np.mean([_.mean() for _ in res_1]) # 12 = 5 * 1 + 7
res_1_scale = np.mean([_.stddev() for _ in res_1]) # 0.76631665 = 1e-3 + tf.math.softplus(0.01 * (6 + 8))
res_1_loc_std = np.std([_.mean() for _ in res_1]) # 1.3837872 = np.sqrt(1 + 1)
res_1_scale_std =  np.std([_.stddev() for _ in res_1]) 
# 0.0071263276 = np.std([1e-3 + tf.math.softplus(0.01 * np.random.normal(loc=14, scale=np.sqrt(2))) for _ in range(1000)])

####################################################

posterior_mean_field = partial(posterior_mean_field_with_initializer, initializer=tf.keras.initializers.Constant([0, 0, 0, 0, 0, 0, 0, 0, 5., 6, 7, 8]))
prior_trainable = partial(prior_trainable_with_initializer, initializer="zero")

model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(2, )),
    
  tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1/2),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
])
n_mc = 1000

res_1 = [model(tf.constant([[0, 1]], dtype=tf.float32)) for _ in range(n_mc)]
res_1_loc = np.mean([_.mean() for _ in res_1]) # 0 = 5 * 0 + 7 * 0
res_1_scale = np.mean([_.stddev() for _ in res_1]) # 0.695651 = 1e-3 + tf.math.softplus(0.01 * 0)
res_1_loc_std = np.std([_.mean() for _ in res_1]) # 8.946621 = np.sqrt(5**2 + 7**2)
res_1_scale_std =  np.std([_.stddev() for _ in res_1]) 
# 0.0538057 = np.std([1e-3 + tf.math.softplus(0.01 * np.random.normal(loc=0, scale=np.sqrt(6**2 + 8 ** 2))) for _ in range(1000)])

res_2 = [model(tf.constant([[0, 2]], dtype=tf.float32)) for _ in range(n_mc)]
res_2_loc = np.mean([_.mean() for _ in res_2]) # 0 = 5 * 0 + 7 * 0
res_2_scale = np.mean([_.stddev() for _ in res_2]) # 0.695651 = 1e-3 + tf.math.softplus(0.01 * 0)
res_2_loc_std = np.std([_.mean() for _ in res_2]) # 12.206555615733702= np.sqrt((5 * 2)**2 + 7**2)
res_2_scale_std =  np.std([_.stddev() for _ in res_2]) 
# 0.0771139 = np.std([1e-3 + tf.math.softplus(0.01 * np.random.normal(loc=0, scale=np.sqrt(12**2 + 8 ** 2))) for _ in range(1000)])