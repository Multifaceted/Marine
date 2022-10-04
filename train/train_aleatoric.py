import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from util.load_data import data_piepline
import tensorflow as tf
import tensorflow_probability as tfp
import pickle

tfd = tfp.distributions

method = "polynomial"
order = 5
save_weights_to = "/home/3068020/Marine/checkpoints/aleatoric_seed0"
save_history_to = "/home/3068020/Marine/history/aleatoric_seed0"
n_epochs = 6000
seed = 0
CTD_Ossigeno_Conducibilita_df = data_piepline(method=method, data_path="../data", resample=False, order=order)

shape, n_vars = CTD_Ossigeno_Conducibilita_df.shape

###################################################################################################################


model_aleatoric = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_vars-2, )),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dense(2, use_bias=True),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
])

model_aleatoric.compile(optimizer="Adam", loss="mse", metrics=["mae"])
tf.keras.utils.set_random_seed(seed)
history = model_aleatoric.fit(CTD_Ossigeno_Conducibilita_df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Ossigeno(mg/l)_CTD"]], CTD_Ossigeno_Conducibilita_df[["Ossigeno(mg/l)_Ossigeno"]], batch_size=shape, epochs=n_epochs)
model_aleatoric.save_weights(save_weights_to)
with open(save_history_to, 'wb') as file_pi:
       pickle.dump(history.history, file_pi)