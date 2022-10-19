import partial
import numpy as np

scale_CTD_temp = 0.023
scale_CTD_press = 0.24
relative_scale_CTD_oxyg = 0.02

scale_node1_temp = 0.15
scale_node1_press = 2
relative_scale_node1_oxyg = 0.052

augFunc_CTD_temp = partial(np.random.normal, scale=scale_CTD_temp)
augFunc_CTD_press = partial(np.random.normal, scale=scale_CTD_press)
augFunc_CTD_oxyg = lambda x: x * np.random.uniform(low=1-relative_scale_CTD_oxyg, high=1+relative_scale_CTD_oxyg)

augFunc_node1_temp = lambda x: x + np.random.uniform(low=-scale_node1_temp, high=scale_node1_temp)
augFunc_node1_press = lambda x: x + np.random.uniform(low=-scale_node1_press, high=scale_node1_press)
augFunc_node1_oxyg = lambda x: x * np.random.uniform(low=-relative_scale_node1_oxyg, high=relative_scale_node1_oxyg)

def resample(df, n=1):
    import pandas as pd
    
    resampled_df = pd.DataFrame([
        df["Temperatura(°C)_CTD"].apply(lambda x: augFunc_CTD_temp(x)),
                                     df["Pressione(db)_CTD"].apply(lambda x: augFunc_CTD_press(x)),
                                     df["Ossigeno(mg/l)_CTD"].apply(lambda x: augFunc_CTD_oxyg(x)),

                                     df["Temperatura(°C)_Ossigeno"].apply(lambda x: augFunc_node1_temp(x)),
                                     df["Pressione(db)_Ossigeno"].apply(lambda x: augFunc_node1_press(x)),
                                     df["Ossigeno(mg/l)_Ossigeno"].apply(lambda x: augFunc_node1_oxyg(x))]).T
    # resampled_df = pd.concat(resampled_df_ls, axis=0, ignore_index=True)
    return resampled_df[["Temperatura(°C)_CTD", "Pressione(db)_CTD", "Ossigeno(mg/l)_CTD", "Temperatura(°C)_Ossigeno", "Pressione(db)_Ossigeno"]], resampled_df[["Ossigeno(mg/l)_Ossigeno"]]