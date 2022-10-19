import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from functools import partial
from util.model import init_model_stochastic, init_model_aleatoric
from util.load_data import data_pipeline, data_pipeline_split
from util.prior_posterior import  posterior_mean_field_with_initializer, prior_trainable_with_initializer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

method = "slinear"
order = 5

CTD_Ossigeno_Conducibilita_df = data_pipeline(method=method, data_path="../data", resample=False, order=order)

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

posterior_mean_field = partial(posterior_mean_field_with_initializer, initializer="zero")
prior_trainable = partial(prior_trainable_with_initializer, initializer="zero")

path = "/home/3068020/Marine/history/stochastic_seed48_slinear"
model_MF = init_model_stochastic(n_inputs=7, posterior=posterior_mean_field, prior=prior_trainable, kl_weight=1./2)

model_MF.load_weights(os.path.join(path, "weights"))

m, s, loss_avg = plot_average(model_MF, CTD_Ossigeno_Conducibilita_df, 10, "without initialization", path=path, save_fig=True)
# np.mean((CTD_Ossigeno_Conducibilita_df["Ossigeno(mg/l)_Ossigeno"] - m) ** 2)

# model = init_model_stochastic(n_inputs=7, posterior=posterior_mean_field, prior=prior_trainable, kl_weight=1./2)

model_aleatoric = init_model_aleatoric(n_inputs=7)

path2 = "/home/3068020/Marine/history/aleatoric_seed48_slinear"
model_aleatoric.load_weights(os.path.join(path2, "weights"))

m, s, loss_avg = plot_average(model_aleatoric, CTD_Ossigeno_Conducibilita_df, 1, "only aleatoric", path=path2, save_fig=True)
# np.mean((CTD_Ossigeno_Conducibilita_df["Ossigeno(mg/l)_Ossigeno"] - m) ** 2)


path3 = "/home/3068020/Marine/history/stochastic_seed48_slinear_initialized"
model_MF_initialized = init_model_stochastic(n_inputs=7, posterior=posterior_mean_field, prior=prior_trainable, kl_weight=1./2)

model_MF_initialized.load_weights(os.path.join(path3, "weights"))
m, s, loss_avg = plot_average(model_MF_initialized, CTD_Ossigeno_Conducibilita_df, 10, "with initialization", path=path3, save_fig=True)

def plot_average(model, df, n_predictions, description=None, save_fig=False, path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.clf()
    plt.figure(figsize=(10, 7), dpi=100)
    x_tst = df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Conducibilita'(mS/cm)_Conducibilita", "Ossigeno(mg/l)_CTD"]].to_numpy()
    yhats = [model(x_tst) for _ in range(n_predictions)]
    
    m = np.mean([np.squeeze(yhat.mean()) for yhat in yhats], axis=0)
    s = np.mean([np.squeeze(yhat.stddev()) for yhat in yhats], axis=0)
    loss_avg = np.mean((df["Ossigeno(mg/l)_Ossigeno"] - m) ** 2)
    plt.plot(df["Time_rounded"], m, 'r', label='ensemble means')
    plt.plot(df["Time_rounded"], m + 2 * s, 'g', linewidth=0.5, label='ensemble means + 2 ensemble stdev', color="green");
    plt.plot(df["Time_rounded"], m - 2 * s, 'g', linewidth=0.5, label='ensemble means - 2 ensemble stdev', color="green");
    plt.plot(df["Time_rounded"], df["Ossigeno(mg/l)_Ossigeno"], label="True", color="blue")
    plt.legend()
    plt.title("aleatoric uncertainty \naveraged by " + str(n_predictions) + " runs " + description + "\nAvg Loss: " + str(round(loss_avg, 5)))
    plt.xlabel("Date")
    plt.ylabel("Oxygen (mg/l)")
    if save_fig:
        plt.savefig(os.path.join(path, "fig.jpg"))
    else:
        plt.show()
    
    return m, s, loss_avg

###############################################################################
method = "slinear"
order = 5

CTD_Ossigeno_Conducibilita_train_df, CTD_Ossigeno_Conducibilita_test_df = data_pipeline_split(method=method, seed=0, data_path="../data", resample=False, order=order)

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

posterior_mean_field = partial(posterior_mean_field_with_initializer, initializer="zero")
prior_trainable = partial(prior_trainable_with_initializer, initializer="zero")

path = "/home/3068020/Marine/history/stochastic_seed24_slinear_traintest"
model_MF = init_model_stochastic(n_inputs=8, posterior=posterior_mean_field, prior=prior_trainable, kl_weight=1./2)

model_MF.load_weights(os.path.join(path, "weights"))

m, s, loss_avg = plot_average(model_MF, CTD_Ossigeno_Conducibilita_test_df.sort_values(by=["Time_rounded"]), 500, "without initialization", path=path, save_fig=True)
# np.mean((CTD_Ossigeno_Conducibilita_df["Ossigeno(mg/l)_Ossigeno"] - m) ** 2)

method = "slinear"
order = 5

CTD_Ossigeno_Conducibilita_train_df, CTD_Ossigeno_Conducibilita_test_df = data_pipeline_split(method=method, seed=0, data_path="../data", resample=False, order=order)


path = "/home/3068020/Marine/history/stochastic_seed24_slinear_initialized_traintest"
model_MF_initialized = init_model_stochastic(n_inputs=8, posterior=posterior_mean_field, prior=prior_trainable, kl_weight=1./2)

model_MF_initialized.load_weights(os.path.join(path, "weights"))

m, s, loss_avg = plot_average(model_MF_initialized, CTD_Ossigeno_Conducibilita_test_df.sort_values(by=["Time_rounded"]), 500, "with initialization", path=path, save_fig=True)

#####################################################################################

model_aleatoric = init_model_aleatoric(n_inputs=8)

path2 = "/home/3068020/Marine/history/aleatoric_seed24_slinear_traintest"
model_aleatoric.load_weights(os.path.join(path2, "weights"))

m, s, loss_avg = plot_average(model_aleatoric, CTD_Ossigeno_Conducibilita_test_df.sort_values(by=["Time_rounded"]), 1, "only aleatoric", path=path2, save_fig=True)