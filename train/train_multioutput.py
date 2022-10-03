import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_inputs, )),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dense(2, use_bias=True),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
])