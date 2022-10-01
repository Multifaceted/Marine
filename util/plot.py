import matplotlib.pyplot as plt
import numpy as np

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