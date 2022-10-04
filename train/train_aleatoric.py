import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from util.load_data import data_piepline
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

method = "polynomial"
order = 5
save_weights_to = "/home/3068020/Marine/checkpoints/stochastic"
save_history_to = "/home/3068020/Marine/history/stochastic"
seed = 42
CTD_Ossigeno_Conducibilita_df = data_piepline(method=method, data_path="../data", resample=False, order=order)

shape, n_vars = CTD_Ossigeno_Conducibilita_df.shape

###################################################################################################################


model_multioutput = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_vars-2, )),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dense(2, use_bias=True),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
])

model_multioutput.compile(optimizer="Adam", loss="mse", metrics=["mae"])
tf.keras.utils.set_random_seed(42)
model_multioutput.fit(CTD_Ossigeno_Conducibilita_df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Ossigeno(mg/l)_CTD"]], CTD_Ossigeno_Conducibilita_df[["Ossigeno(mg/l)_Ossigeno"]], batch_size=shape, epochs=6000)
model_multioutput.save_weights("/home/3068020/Marine/saved_model/multioutput")