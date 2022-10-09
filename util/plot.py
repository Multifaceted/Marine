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
import matplotlib.pyplot as plt
import numpy as np

method = "polynomial"
order = 5

CTD_Ossigeno_Conducibilita_df = data_piepline(method=method, data_path="../data", resample=False, order=order)

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

posterior_mean_field = partial(posterior_mean_field_with_initializer, initializer="zero")
prior_trainable = partial(prior_trainable_with_initializer, initializer="zero")

model_MF = init_model_stochastic(n_inputs=7, posterior=posterior_mean_field, prior=prior_trainable, kl_weight=1./2)

model_MF.load_weights("/home/3068020/Marine/checkpoints/stochastic_initialized_seed24_48_delta1_slinear")

m, s = plot_average(model_MF, CTD_Ossigeno_Conducibilita_df, 10)
np.mean((CTD_Ossigeno_Conducibilita_df["Ossigeno(mg/l)_Ossigeno"] - m) ** 2)

model = init_model_stochastic(n_inputs=7, posterior=posterior_mean_field, prior=prior_trainable, kl_weight=1./2)

model.load_weights("/home/3068020/Marine/checkpoints/stochastic_seed48_slinear")

m, s = plot_average(model, CTD_Ossigeno_Conducibilita_df, 10)
np.mean((CTD_Ossigeno_Conducibilita_df["Ossigeno(mg/l)_Ossigeno"] - m) ** 2)


def plot_average(model, df, n_predictions):
    plt.clf()

    x_tst = df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Ossigeno(mg/l)_CTD"]].to_numpy()
    yhats = [model(x_tst) for _ in range(n_predictions)]
    avgm = np.zeros_like(x_tst[..., 0])
    m = np.mean([np.squeeze(yhat.mean()) for yhat in yhats], axis=0)
    s = np.mean([np.squeeze(yhat.stddev()) for yhat in yhats], axis=0)
    plt.plot(df["Time_rounded"], m, 'r', label='ensemble means')
    plt.plot(df["Time_rounded"], m + 2 * s, 'g', linewidth=0.5, label='ensemble means + 2 ensemble stdev', color="green");
    plt.plot(df["Time_rounded"], m - 2 * s, 'g', linewidth=0.5, label='ensemble means - 2 ensemble stdev', color="green");
    plt.plot(df["Time_rounded"], df["Ossigeno(mg/l)_Ossigeno"], label="True", color="blue")
    plt.legend()
    plt.title("aleatoric uncertainty")
    plt.show()

    return m, s