import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from pathlib import Path
from util.load_data import data_pipeline_split
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import argparse
import json
from functools import partial
import numpy as np

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

CTD_Ossigeno_Conducibilita_train_df, CTD_Ossigeno_Conducibilita_test_df = data_pipeline_split(method=method, seed=0, data_path="../data", resample=False, order=order)

shape, n_vars = CTD_Ossigeno_Conducibilita_train_df.shape

###################################################################################################################
Path(save_to).mkdir(parents=True, exist_ok=True)

scale_CTD_temp = 0.023
scale_CTD_press = 0.24
relative_scale_CTD_oxyg = 0.02

scale_node1_temp = 0.15
scale_node1_press = 2
relative_scale_node1_oxyg = 0.052

scale_node2_temp = 0.05
scale_node2_press = 2
scale_node2_conduct = 0.05

augFunc_CTD_temp = partial(np.random.normal, scale=scale_CTD_temp)
augFunc_CTD_press = partial(np.random.normal, scale=scale_CTD_press)
augFunc_CTD_oxyg = lambda x: x * np.random.uniform(low=1-relative_scale_CTD_oxyg, high=1+relative_scale_CTD_oxyg)

augFunc_node1_temp = lambda x: x + np.random.uniform(low=-scale_node1_temp, high=scale_node1_temp)
augFunc_node1_press = lambda x: x + np.random.uniform(low=-scale_node1_press, high=scale_node1_press)
augFunc_node1_oxyg = lambda x: x * np.random.uniform(low=-relative_scale_node1_oxyg, high=relative_scale_node1_oxyg)

augFunc_node2_temp = lambda x: x + np.random.uniform(low=-scale_node2_temp, high=scale_node2_temp)
augFunc_node2_press = lambda x: x + np.random.uniform(low=-scale_node2_press, high=scale_node2_press)
augFunc_node2_conduct = lambda x: x + np.random.uniform(low=-scale_node2_conduct, high=scale_node2_conduct)

def resample(df, n=1):
    import pandas as pd
    
    while True:
        resampled_df = pd.DataFrame([
            df["Temperatura(°C)_CTD"].apply(lambda x: augFunc_CTD_temp(x)),
            df["Pressione(db)_CTD"].apply(lambda x: augFunc_CTD_press(x)),
            df["Ossigeno(mg/l)_CTD"].apply(lambda x: augFunc_CTD_oxyg(x)),

            df["Temperatura(°C)_Ossigeno"].apply(lambda x: augFunc_node1_temp(x)),
            df["Pressione(db)_Ossigeno"].apply(lambda x: augFunc_node1_press(x)),
            df["Ossigeno(mg/l)_Ossigeno"].apply(lambda x: augFunc_node1_oxyg(x)),
            
            df["Temperatura(°C)_Conducibilita"].apply(lambda x: augFunc_node2_temp(x)),
            df["Pressione(db)_Conducibilita"].apply(lambda x: augFunc_node2_press(x)),
            df["Conducibilita'(mS/cm)_Conducibilita"].apply(lambda x: augFunc_node2_conduct(x))
            ]).T
        # resampled_df = pd.concat(resampled_df_ls, axis=0, ignore_index=True)
        yield resampled_df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Conducibilita'(mS/cm)_Conducibilita", "Ossigeno(mg/l)_CTD"]].to_numpy().tolist(), resampled_df[["Ossigeno(mg/l)_Ossigeno"]].to_numpy().tolist()

gen = partial(resample, df=CTD_Ossigeno_Conducibilita_train_df)

train_gen = tf.data.Dataset.from_generator(gen, output_signature=(
              tf.TensorSpec(shape=(None, 8), dtype=tf.float32),
              tf.TensorSpec(shape=(None, 1), dtype=tf.float32)))



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

seed = 100
n_epochs = 10
model_aleatoric.compile(optimizer="Adam", loss="mse", metrics=["mae"])
tf.keras.utils.set_random_seed(seed)
# history = model_aleatoric.fit(CTD_Ossigeno_Conducibilita_train_df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Conducibilita'(mS/cm)_Conducibilita", "Ossigeno(mg/l)_CTD"]], CTD_Ossigeno_Conducibilita_train_df[["Ossigeno(mg/l)_Ossigeno"]], batch_size=shape, epochs=n_epochs)
history = model_aleatoric.fit(train_gen, steps_per_epoch=1, epochs=n_epochs, verbose=1)
model_aleatoric.save_weights(os.path.join(save_to, "weights"))

with open(os.path.join(save_to, "history"), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

with open(os.path.join(save_to, "params"), 'w') as outfile:
    json.dump(params, outfile, indent=4)