import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from functools import partial
from util.model import init_model_stochastic
from util.load_data import data_piepline
from util.prior_posterior import  posterior_mean_field_with_initializer, prior_trainable_with_initializer
import tensorflow as tf
import pickle

method = "polynomial"
order = 5
save_weights_to = "/home/3068020/Marine/checkpoints/stochastic"
save_history_to = "/home/3068020/Marine/history/stochastic"
seed = 42
CTD_Ossigeno_Conducibilita_df = data_piepline(method=method, data_path="../data", resample=False, order=order)


shape, n_vars = CTD_Ossigeno_Conducibilita_df.shape

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

posterior_mean_field = partial(posterior_mean_field_with_initializer, initializer="zero")
prior_trainable = partial(prior_trainable_with_initializer, initializer="zero")

###################################################################################################################
model_MF = init_model_stochastic(n_inputs=n_vars-2, posterior=posterior_mean_field, prior=prior_trainable, kl_weight=1./shape)
model_MF.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
tf.keras.utils.set_random_seed(seed)
history = model_MF.fit(CTD_Ossigeno_Conducibilita_df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Ossigeno(mg/l)_CTD"]], CTD_Ossigeno_Conducibilita_df[["Ossigeno(mg/l)_Ossigeno"]], batch_size=shape, epochs=3000)

model_MF.save_weights(save_weights_to)
with open(save_history_to, 'wb') as file_pi:
       pickle.dump(history.history, file_pi)