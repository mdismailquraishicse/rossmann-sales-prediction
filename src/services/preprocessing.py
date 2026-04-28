import numpy as np
import pandas as pd
from src.utils.basics_info import extend_date


class Preprocessing:



    def __init__(self):

        pass


    def merge_and_transform_dfs(self, df_rossmann:pd.DataFrame, df_store:pd.DataFrame, is_train:bool):
    
        print(f"shape rossmann: {df_rossmann.shape}")
        print(f"shape store: {df_store.shape}")
        df = df_rossmann.merge(df_store, how="left", on="Store")
        print(f"shape after merge: {df.shape}")
        cols_in_test = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
        if is_train:
            cols_in_test.append("Sales")
        df = df[cols_in_test] # Keep only those cols which are available in test only.
        print(f"shape with test cols only: {df.shape}")
        print("converting columns to lowercase...")
        df.columns = [col.lower().strip() for col in df.columns]
        return df
    

    def filter_data(self, df:pd.DataFrame, cols2drop:list = ["open"]):
    
        print(f"shape: {df.shape}")
        df = df[df["open"]==1]
        print(f"shape: {df.shape}")
        df = df.drop(cols2drop, axis=1)
        print(f"shape: {df.shape}")
        return df
    

    def data_preprocessing(self, df:pd.DataFrame):
    
        print(f"shape: {df.shape}")
        df["stateholiday"] = df["stateholiday"].astype(str)
        df["schoolholiday"] = df["schoolholiday"].astype(int)
        df = extend_date(df, "date")
        print(f"shape: {df.shape}")
        return df
