from util.load_data import read_pipeline
from util.interpolate import interpolate
from util.model import init_model
from util.prior_posterior import  posterior_mean_field, prior_trainable
from util.plot import plot_average
import tensorflow as tf

method = "polynomial"
order = 5

Ossigeno_without_na_sub_df, Conducibilita_without_na_sub_df, CTD_without_na_sub_df = read_pipeline(resample=False)

CTD_Ossigeno_Conducibilita_df = Ossigeno_without_na_sub_df.merge(Conducibilita_without_na_sub_df, how="left", on="Time_rounded", suffixes=("_Ossigeno", "_Conducibilita")).dropna().merge(CTD_without_na_sub_df, on="Time_rounded", how="left", suffixes=("", "_CTD"))

CTD_Ossigeno_Conducibilita_df = interpolate(CTD_Ossigeno_Conducibilita_df, ["Temperatura(°C)_CTD", "Pressione(db)_CTD", "Ossigeno(mg/l)_CTD"], method=method, order=order)

CTD_Ossigeno_Conducibilita_df = CTD_Ossigeno_Conducibilita_df[["Time_rounded", "Ossigeno(mg/l)_Ossigeno", "Ossigeno(mg/l)_CTD", "Temperatura(°C)_Ossigeno", "Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Pressione(db)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita"]].dropna()


shape = CTD_Ossigeno_Conducibilita_df.shape[0]

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

model_MF = init_model(n_inputs=CTD_Ossigeno_Conducibilita_df.shape[1]-2, posterior=posterior_mean_field, prior=prior_trainable, kl_weight=1./shape)
model_MF.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
model_MF.fit(CTD_Ossigeno_Conducibilita_df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Ossigeno(mg/l)_CTD"]], CTD_Ossigeno_Conducibilita_df[["Ossigeno(mg/l)_Ossigeno"]], batch_size=shape, epochs=3000)

distri_output = model_MF(CTD_Ossigeno_Conducibilita_df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Ossigeno(mg/l)_CTD"]].to_numpy())
                        
n_predictions = 10

plot_average(model_MF, CTD_Ossigeno_Conducibilita_df, n_predictions)

tf.keras.utils.set_random_seed(42)
model_deterministic = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(CTD_Ossigeno_Conducibilita_df.shape[1]-2, )),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1)
])
model_deterministic.compile(optimizer="Adam", loss="mse", metrics=["mae"])

model_deterministic.fit(CTD_Ossigeno_Conducibilita_df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Ossigeno(mg/l)_CTD"]], CTD_Ossigeno_Conducibilita_df[["Ossigeno(mg/l)_Ossigeno"]], batch_size=shape, epochs=6000)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

n_predictions=1
df = CTD_Ossigeno_Conducibilita_df
x_tst = df[["Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Temperatura(°C)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Pressione(db)_Ossigeno", "Ossigeno(mg/l)_CTD"]].to_numpy()
yhats = [model_deterministic.predict(x_tst) for _ in range(n_predictions)]
m = np.mean([np.squeeze(yhat) for yhat in yhats], axis=0)
plt.plot(df["Time_rounded"], m, 'r', label='ensemble means')
plt.plot(df["Time_rounded"], df["Ossigeno(mg/l)_Ossigeno"], label="True", color="blue")
plt.legend()
plt.title("aleatoric uncertainty")
plt.show()

# TODO find the order of the weights using a toy model
# TODO Monte Carlo As Bayesian Approximation
model_deterministic = tf.keras.models.load_model('saved_model/test')
model_MF =  init_model(n_inputs=CTD_Ossigeno_Conducibilita_df.shape[1]-2, posterior=posterior_mean_field, prior=prior_trainable, kl_weight=1./shape)
model_MF.load_weights("./checkpoints/MF")

model = tf.keras.Sequential([
  tf.keras.layers.Input((1,)),
      
  tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1/2),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                          scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
    ])

model2 = tf.keras.Sequential([
  tf.keras.layers.Input((1,)),
  tf.keras.layers.Dense(2)      

    ])