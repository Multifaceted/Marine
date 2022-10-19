import matplotlib.pyplot as plt
import pickle

path = "/home/3068020/Marine/history/multioutput_initialized"
with open(path, 'rb') as file_pi:
    history = pickle.load(file_pi)

plt.plot(history["loss"])

import matplotlib.pyplot as plt
import pickle

path = "/home/3068020/Marine/history/multioutput"
with open(path, 'rb') as file_pi:
    history = pickle.load(file_pi)
