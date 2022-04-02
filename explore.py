import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

data_path = "data"

file1 = "TBaia_01m-Conducibilita.txt"
file2 = "TBaia_01m-CTD.txt"
file3 = "TBaia_01m-Ossigeno.txt"
file4 = "TBaia_01m-Winkler.txt"

Conducibilita_raw_df = pd.read_csv(os.path.join(data_path, file1), encoding='cp1252', header=None, skiprows=11)
Conducibilita_raw_df = Conducibilita_raw_df.squeeze().str.strip().apply(lambda x: re.sub("\s+", ",", x)).str.split(",", expand=True)
colNames = Conducibilita_raw_df.iloc[0, 1:].apply(lambda x: re.split(r"[(\\)\'_]", x)[0])
Conducibilita_raw_df = Conducibilita_raw_df.iloc[1:, :]
Conducibilita_raw_df.dropna(axis=1, inplace=True)
Conducibilita_raw_df.columns = colNames

def convertTime(x):
    try:
        return pd.to_datetime(x["Data"] + "/" + x["Ora"], format="%d/%m/%Y/%H:%M:%S")
    except:
        return -1

Conducibilita_raw_df["Time"] = Conducibilita_raw_df[["Data", "Ora"]].apply(lambda x: convertTime(x), axis=1)
Conducibilita_raw_df.info()

CTD_raw_df = pd.read_csv(os.path.join(data_path, file2), encoding='cp1252', header=None, skiprows=15)
CTD_raw_df = CTD_raw_df.squeeze().str.strip().apply(lambda x: re.sub("\s+", ",", x)).str.split(",", expand=True)
colNames = CTD_raw_df.iloc[0, :].apply(lambda x: re.split(r"[(\\)\'_]", x)[0])
CTD_raw_df.columns = colNames
CTD_raw_df = CTD_raw_df.iloc[1:, :]
CTD_raw_df.info()

Ossigeno_raw_df = pd.read_csv(os.path.join(data_path, file3), encoding='cp1252', header=None, skiprows=11)
Ossigeno_raw_df = Ossigeno_raw_df.squeeze().str.strip().apply(lambda x: re.sub("\s+", ",", x)).str.split(",", expand=True)
colNames = Ossigeno_raw_df.iloc[0, :].apply(lambda x: re.split(r"[(\\)\'_]", x)[0])
Ossigeno_raw_df.columns = colNames
Ossigenoraw_df = Ossigeno_raw_df.iloc[1:, :]
Ossigeno_raw_df.info()

Ossigeno_raw_df = pd.read_csv(os.path.join(data_path, file3), encoding='cp1252', header=None, skiprows=11)
Ossigeno_raw_df = Ossigeno_raw_df.squeeze().str.strip().apply(lambda x: re.sub("\s+", ",", x)).str.split(",", expand=True)
colNames = Ossigeno_raw_df.iloc[0, :].apply(lambda x: re.split(r"[(\\)\'_]", x)[0])
Ossigeno_raw_df.columns = colNames
Ossigenoraw_df = Ossigeno_raw_df.iloc[1:, :]
Ossigeno_raw_df.info()

