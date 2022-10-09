import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def init_model_stochastic(n_inputs, posterior, prior, kl_weight):
    """
    Measure both epistemic uncertainty and aleatoric uncertainty.
    The number of parameters in the last-but-one weight layer is equal to 32 (previous neurons) * 2 (mean + stddev) + 2 (bias)E
    """
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_inputs, )),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
        
    tfp.layers.DenseVariational(1 + 1, posterior, prior, kl_weight=kl_weight),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
    ])

    return model

def init_model_aleatoric(n_inputs):
    """
    Model that only measure aleatoric uncertainty. Similar to softmax output in case of classification.
    The estimated result can be used for initialization stochastic model.
    """
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
    return model