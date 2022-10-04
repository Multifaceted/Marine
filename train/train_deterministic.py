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

model_deterministic.save("/home/3068020/Marine/saved_model/test")