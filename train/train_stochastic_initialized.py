import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from functools import partial
from util.model import init_model_stochastic, init_model_aleatoric
from util.load_data import data_piepline
from util.prior_posterior import  posterior_mean_field_with_initializer, prior_trainable_with_initializer
import tensorflow as tf
import numpy as np
import pickle

method = "polynomial"
order = 5
save_weights_to = "/home/3068020/Marine/checkpoints/stochastic_initialized"
save_history_to = "/home/3068020/Marine/history/stochastic_initialized"
seed = 42
CTD_Ossigeno_Conducibilita_df = data_piepline(method=method, data_path="../data", resample=False, order=order)

shape = CTD_Ossigeno_Conducibilita_df.shape[0]

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

###################################################################################################################
model_multioutput = init_model_aleatoric(n_inputs=7)
model_multioutput.load_weights("/home/3068020/Marine/saved_model/multioutput")
weights_initializer = np.concatenate([model_multioutput.weights[-2].numpy().reshape(-1, ), model_multioutput.weights[-1].numpy().reshape(-1, )], axis=0)
delta = 0.1
stds_initializer = delta * np.abs(weights_initializer)

posterior_mean_field = partial(posterior_mean_field_with_initializer, initializer=tf.keras.initializers.Constant(np.concatenate([weights_initializer, stds_initializer], axis=0)))
prior_trainable = partial(prior_trainable_with_initializer, initializer=tf.keras.initializers.Constant(weights_initializer))

###################################################################################################################
model_MF =  init_model_stochastic(n_inputs=CTD_Ossigeno_Conducibilita_df.shape[1]-2, posterior=posterior_mean_field, prior=prior_trainable, kl_weight=1./shape)
model_MF

model_MF.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
tf.keras.utils.set_random_seed(seed)
history = model_MF.fit(CTD_Ossigeno_Conducibilita_df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Ossigeno(mg/l)_CTD"]], CTD_Ossigeno_Conducibilita_df[["Ossigeno(mg/l)_Ossigeno"]], batch_size=shape, epochs=3000)

model_MF.save_weights(save_weights_to)
with open(save_history_to, 'wb') as file_pi:
       pickle.dump(history.history, file_pi)
        