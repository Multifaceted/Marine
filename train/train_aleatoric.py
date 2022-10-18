import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pathlib import Path
from util.load_data import data_pipeline
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import argparse
import json


tfd = tfp.distributions

parser = argparse.ArgumentParser(description='Parameter')
parser.add_argument("-p", "--param", type=str, required=True)

args = parser.parse_args()

with open(args.param, 'r') as f:
  params = json.load(f)

method = params["method"]
order = params["order"]
save_to = params["save_to"]
n_epochs = params["n_epochs"]
seed = params["seed"]

CTD_Ossigeno_Conducibilita_df = data_pipeline(method=method, data_path="../data", resample=False, order=order)

shape, n_vars = CTD_Ossigeno_Conducibilita_df.shape

###################################################################################################################
Path(save_to).mkdir(parents=True, exist_ok=True)

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
model_aleatoric.save_weights(os.path.join(save_to, "weights"))

with open(os.path.join(save_to, "history"), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

with open(os.path.join(save_to, "params"), 'w') as outfile:
    json.dump(params, outfile, indent=4)