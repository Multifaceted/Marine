import pandas as pd
import numpy as np
from util.utils import round_time
from util.utils import interpolate
import inspect
import sys
import os
import re

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

def read_pipeline(data_path="data", resample=False, **kwargs):
    file_Conduvibilita = "TBaia_01m-Conducibilita'.txt"
    file_CTD = "TBaia_01m-CTD.txt"
    file_Ossigeno = "TBaia_01m-Ossigeno.txt"
    file_Winkler = "TBaia_01m-Winkler.txt"

    def convertTime(x):
        try:
            return pd.to_datetime(x["Data"] + "/" + x["Ora(UTC)"], format="%d/%m/%Y/%H:%M:%S")
        except:
            return -1

    ################### Conducibilita ###################
    Conducibilita_raw_df = pd.read_csv(os.path.join(data_path, file_Conduvibilita), encoding='cp1252', header=None, skiprows=11)
    Conducibilita_raw_df.iloc[0, 0] = re.sub("#", "", Conducibilita_raw_df.iloc[0, 0]).strip()
    Conducibilita_raw_df = Conducibilita_raw_df.squeeze().str.strip().apply(lambda x: re.sub("\s+", ",", x)).str.split(",", expand=True)
    # colNames = Conducibilita_raw_df.iloc[0, :].apply(lambda x: re.split(r"[(\\)\'_]", x)[0])
    colNames = Conducibilita_raw_df.iloc[0, :]

    Conducibilita_raw_df.columns = colNames
    Conducibilita_raw_df = Conducibilita_raw_df.iloc[1:, :]

    for j in range(2, Conducibilita_raw_df.shape[1]):
        Conducibilita_raw_df.iloc[:, j] = Conducibilita_raw_df.iloc[:, j].astype(np.float32)

    Conducibilita_raw_df["Time"] = Conducibilita_raw_df[["Data", "Ora(UTC)"]].apply(lambda x: convertTime(x), axis=1)
    Conducibilita_raw_df["Data"] = Conducibilita_raw_df["Time"].dt.date
    Conducibilita_raw_df["Ora(UTC)"] =  Conducibilita_raw_df["Time"].dt.time

    ################### CTD ###################

    CTD_raw_df = pd.read_csv(os.path.join(data_path, file_CTD), encoding='cp1252', header=None, skiprows=15)
    CTD_raw_df.iloc[0, 0] = re.sub("#", "", CTD_raw_df.iloc[0, 0]).strip()
    CTD_raw_df = CTD_raw_df.squeeze().str.strip().apply(lambda x: re.sub("\s+", ",", x)).str.split(",", expand=True)
    # colNames = CTD_raw_df.iloc[0, :].apply(lambda x: re.split(r"[(\\)\'_]", x)[0])
    colNames = CTD_raw_df.iloc[0, :]

    CTD_raw_df.columns = colNames
    CTD_raw_df = CTD_raw_df.iloc[1:, :]

    for j in range(2, CTD_raw_df.shape[1]):
        CTD_raw_df.iloc[:, j] = CTD_raw_df.iloc[:, j].astype(np.float32)

    CTD_raw_df["Time"] = CTD_raw_df[["Data", "Ora(UTC)"]].apply(lambda x: convertTime(x), axis=1)
    CTD_raw_df["Data"] = CTD_raw_df["Time"].dt.date
    CTD_raw_df["Ora(UTC)"] =  CTD_raw_df["Time"].dt.time

    ################### Ossigeno ###################
    Ossigeno_raw_df = pd.read_csv(os.path.join(data_path, file_Ossigeno), encoding='cp1252', header=None, skiprows=11)
    Ossigeno_raw_df.iloc[0, 0] = re.sub("#", "", Ossigeno_raw_df.iloc[0, 0]).strip()
    Ossigeno_raw_df = Ossigeno_raw_df.squeeze().str.strip().apply(lambda x: re.sub("\s+", ",", x)).str.split(",", expand=True)
    # colNames = Ossigeno_raw_df.iloc[0, :].apply(lambda x: re.split(r"[(\\)\'_]", x)[0])
    colNames = Ossigeno_raw_df.iloc[0, :]

    Ossigeno_raw_df.columns = colNames
    Ossigeno_raw_df = Ossigeno_raw_df.iloc[1:, :]

    for j in range(2, Ossigeno_raw_df.shape[1]):
        Ossigeno_raw_df.iloc[:, j] = Ossigeno_raw_df.iloc[:, j].astype(np.float32)

    Ossigeno_raw_df["Time"] = Ossigeno_raw_df[["Data", "Ora(UTC)"]].apply(lambda x: convertTime(x), axis=1)
    Ossigeno_raw_df["Data"] = Ossigeno_raw_df["Time"].dt.date
    Ossigeno_raw_df["Ora(UTC)"] =  Ossigeno_raw_df["Time"].dt.time

    ################### Winkler ###################
    Winkler_raw_df = pd.read_csv(os.path.join(data_path, file_Winkler), encoding='cp1252', header=None, skiprows=10)
    Winkler_raw_df.iloc[0, 0] = re.sub("#", "", Winkler_raw_df .iloc[0, 0]).strip()
    Winkler_raw_df = Winkler_raw_df.squeeze().str.strip().apply(lambda x: re.sub("\s+", ",", x)).str.split(",", expand=True)
    colNames = Winkler_raw_df.iloc[0, :]
    # colNames = Winkler_raw_df.iloc[0, :].apply(lambda x: re.split(r"[(\\)\'_]", x)[0])

    Winkler_raw_df.columns = colNames
    Winkler_raw_df = Winkler_raw_df.iloc[1:, :]

    for j in range(2,  Winkler_raw_df.shape[1]):
        Winkler_raw_df.iloc[:, j] = Winkler_raw_df.iloc[:, j].astype(np.float32)

    Winkler_raw_df["Time"] = Winkler_raw_df[["Data", "Ora(UTC)"]].apply(lambda x: convertTime(x), axis=1)
    Winkler_raw_df["Data"] = Winkler_raw_df["Time"].dt.date
    Winkler_raw_df["Ora(UTC)"] =  Winkler_raw_df["Time"].dt.time

    ################### Drop NA ###################
    Ossigeno_na_df = Ossigeno_raw_df.where( Ossigeno_raw_df!=-9999, other=None )
    Winkler_na_df = Winkler_raw_df.where( Winkler_raw_df!=-9999, other=None )
    CTD_na_df = CTD_raw_df.where( CTD_raw_df!=-9999, other=None )
    Conducibilita_na_df = Conducibilita_raw_df.where( Conducibilita_raw_df!=-9999, other=None )

    Conducibilita_without_na_df = Conducibilita_na_df.dropna()
    Ossigeno_without_na_df = Ossigeno_na_df.dropna(subset=["Ossigeno(mg/l)"])
    Winkler_without_na_df = Winkler_na_df.dropna()
    CTD_without_na_df = CTD_na_df.dropna(subset=["Ossigeno(mg/l)"])

    ################### Subset DF ###################
    Ossigeno_without_na_sub_df = Ossigeno_without_na_df[["Data", "Ora(UTC)", "Pressione(db)", "Ossigeno(mg/l)", "Temperatura(°C)", "Time"]]
    CTD_without_na_sub_df = CTD_without_na_df[["Data", "Ora(UTC)", "Pressione(db)", "Ossigeno(mg/l)", "Temperatura(°C)", "Time"]]
    Conducibilita_without_na_sub_df = Conducibilita_without_na_df[["Data", "Ora(UTC)", "Pressione(db)", "Conducibilita'(mS/cm)", "Temperatura(°C)", "Time"]]

    ################### Round Time ###################
    Ossigeno_without_na_sub_df["Time_rounded"] = Ossigeno_without_na_sub_df["Time"]
    CTD_without_na_sub_df["Time_rounded"] = CTD_without_na_sub_df["Time"]
    Conducibilita_without_na_sub_df["Time_rounded"] = Conducibilita_without_na_sub_df["Time"]
    Ossigeno_without_na_sub_df["Time_rounded"] = Ossigeno_without_na_sub_df["Time_rounded"].apply(lambda x: round_time(x))
    CTD_without_na_sub_df["Time_rounded"] = CTD_without_na_sub_df["Time_rounded"].apply(lambda x: round_time(x))
    Conducibilita_without_na_sub_df["Time_rounded"] = Conducibilita_without_na_sub_df["Time_rounded"].apply(lambda x: round_time(x))

    Ossigeno_without_na_sub_df.rename({"Ossigeno(mg/l)": "Ossigeno(mg/l)_Ossigeno", "Pressione(db)": "Pressione(db)_Ossigeno", "Temperatura(°C)": "Temperatura(°C)_Ossigeno"}, axis=1, inplace=True)
    Conducibilita_without_na_sub_df.rename({"Conducibilita'(mS/cm)": "Conducibilita'(mS/cm)_Conducibilita", "Pressione(db)": "Pressione(db)_Conducibilita", "Temperatura(°C)": "Temperatura(°C)_Conducibilita"}, axis=1, inplace=True)
    CTD_without_na_sub_df.rename({"Ossigeno(mg/l)": "Ossigeno(mg/l)_CTD", "Pressione(db)": "Pressione(db)_CTD", "Temperatura(°C)": "Temperatura(°C)_CTD"}, axis=1, inplace=True)
    
    return Ossigeno_without_na_sub_df, Conducibilita_without_na_sub_df, CTD_without_na_sub_df

def data_pipeline(method, data_path="../data", **kwargs):
    Ossigeno_without_na_sub_df, Conducibilita_without_na_sub_df, CTD_without_na_sub_df = read_pipeline(data_path=data_path, **kwargs)

    CTD_Ossigeno_Conducibilita_df = Ossigeno_without_na_sub_df.merge(Conducibilita_without_na_sub_df, how="left", on="Time_rounded", suffixes=("_Ossigeno", "_Conducibilita")).dropna().merge(CTD_without_na_sub_df, on="Time_rounded", how="left", suffixes=("", "_CTD"))

    CTD_Ossigeno_Conducibilita_df = interpolate(CTD_Ossigeno_Conducibilita_df, ["Temperatura(°C)_CTD", "Pressione(db)_CTD", "Ossigeno(mg/l)_CTD"], method=method, **kwargs)

    CTD_Ossigeno_Conducibilita_df = CTD_Ossigeno_Conducibilita_df[["Time_rounded", "Ossigeno(mg/l)_Ossigeno", "Ossigeno(mg/l)_CTD", "Temperatura(°C)_Ossigeno", "Temperatura(°C)_CTD", "Temperatura(°C)_Conducibilita", "Pressione(db)_Ossigeno", "Pressione(db)_CTD", "Pressione(db)_Conducibilita", "Conducibilita'(mS/cm)_Conducibilita"]].dropna()

    return CTD_Ossigeno_Conducibilita_df

def data_pipeline_split(method, seed, data_path="../data", **kwargs):
    from sklearn.model_selection import train_test_split

    CTD_Ossigeno_Conducibilita_df = data_pipeline(method=method, data_path=data_path, **kwargs)

    return train_test_split(CTD_Ossigeno_Conducibilita_df, test_size=.2, random_state=seed)