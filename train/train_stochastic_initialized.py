import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from functools import partial
from util.model import init_model_stochastic, init_model_aleatoric
from util.load_data import data_pipeline
from util.prior_posterior import  posterior_mean_field_with_initializer, prior_trainable_with_initializer
import tensorflow as tf
import numpy as np
from pathlib import Path
import pickle
import argparse
import json

parser = argparse.ArgumentParser(description='Parameter')
parser.add_argument("-p", "--param", type=str, required=True)

args = parser.parse_args()



params = {"method": "polynomial",
"order": 5,
"save_to": "/home/3068020/Marine/history/stochastic_initialized_seed24_48_delta1_slinear",
"load_weights_from": "/home/3068020/Marine/history/aleatoric_seed24_slinear",
"n_epochs": 1000,
"seed": 48}

with open(args.param, 'r') as f:
  params = json.load(f)

method = params["method"]
order = params["order"]
save_to = params["save_to"]
load_weights_from = params["load_weights_from"]
n_epochs = params["n_epochs"]
seed = params["seed"]

CTD_Ossigeno_Conducibilita_df = data_pipeline(method=method, data_path="../data", resample=False, order=order)

shape, n_vars = CTD_Ossigeno_Conducibilita_df.shape

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

###################################################################################################################
model_multioutput = init_model_aleatoric(n_vars-2)
model_multioutput.load_weights(os.path.join(load_weights_from, "weights"))
weights_initializer = np.concatenate([model_multioutput.weights[-2].numpy().reshape(-1, ), model_multioutput.weights[-1].numpy().reshape(-1, )], axis=0)
delta = 1E-3
stds_initializer = delta * np.abs(weights_initializer)

posterior_mean_field = partial(posterior_mean_field_with_initializer, initializer=tf.keras.initializers.Constant(np.concatenate([weights_initializer, stds_initializer], axis=0)))
prior_trainable = partial(prior_trainable_with_initializer, initializer=tf.keras.initializers.Constant(weights_initializer))

###################################################################################################################
Path(save_to).mkdir(parents=True, exist_ok=True)

model_MF =  init_model_stochastic(n_inputs=n_vars-2, posterior=posterior_mean_field, prior=prior_trainable, kl_weight=1./shape)
model_MF.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
tf.keras.utils.set_random_seed(seed)
history = model_MF.fit(CTD_Ossigeno_Conducibilita_df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Ossigeno(mg/l)_CTD"]], CTD_Ossigeno_Conducibilita_df[["Ossigeno(mg/l)_Ossigeno"]], batch_size=shape, epochs=n_epochs)

model_MF.save_weights(os.path.join(save_to, "weights"))

with open(os.path.join(save_to, "history"), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

with open(os.path.join(save_to, "params"), 'w') as outfile:
    json.dump(params, outfile, indent=4)