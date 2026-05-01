import sys
import pandas as pd


def extend_date(df:pd.DataFrame, colname:str):

    df[colname] = pd.to_datetime(df[colname])
    df["year"] = df[colname].dt.year
    df["month"] = df[colname].dt.month
    df["day"] = df[colname].dt.day
    df["week_of_year"] = df[colname].dt.isocalendar().week
    return df


class FeatureEngineering:



    def merge_and_transform_dfs(self, df_rossmann:pd.DataFrame, df_store:pd.DataFrame, is_train:bool):
    
        print(f"shape rossmann: {df_rossmann.shape}")
        print(f"shape store: {df_store.shape}")
        df = df_rossmann.merge(df_store, how="left", on="Store")
        print(f"shape after merge: {df.shape}")
        cols_in_test = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
        cols = cols_in_test.copy()
        if is_train:
            cols.append("Sales")
        df = df[cols] # Keep only those cols which are available in test only.
        print(f"shape with test cols only: {df.shape}")
        print("converting columns to lowercase...")
        df.columns = [col.lower().strip() for col in df.columns]
        return df
    
    
    def filter_open_stores(self, df:pd.DataFrame, cols2drop= None):

        if cols2drop is None:
            cols2drop = ["open"]
        print(f"shape: {df.shape}")
        if "open" in df.columns:
            df = df[df["open"]==1]
        print(f"shape: {df.shape}")
        df = df.drop(cols2drop, axis=1)
        print(f"shape: {df.shape}")
        return df


    def extend_date(self, df:pd.DataFrame, colname:str):

        df[colname] = pd.to_datetime(df[colname])
        df["year"] = df[colname].dt.year
        df["month"] = df[colname].dt.month
        df["day"] = df[colname].dt.day
        df["week_of_year"] = df[colname].dt.isocalendar().week
        df = df.drop("date", axis = 1)
        return df
    
    
    def data_preprocessing(self, df:pd.DataFrame):
    
        print(f"shape: {df.shape}")
        df = self.extend_date(df, "date")
        if "year" in df.columns:
            df = df.drop("year", axis = 1)
        print(f"shape: {df.shape}")
        return df



class FeatureTransformer:


    def __init__(self):

        self.num_cols = ["month", "week_of_year"]
        self.scaler = None
        self.final_columns = None
        self.store_mapping = None


    def fit(self, df: pd.DataFrame):

        from sklearn.preprocessing import StandardScaler

        df = df.copy()

        # type conversions (safe here)
        df["stateholiday"] = df["stateholiday"].astype(str)
        df["schoolholiday"] = df["schoolholiday"].astype(int)

        categories = df["store"].astype("category").cat.categories
        self.store_mapping = {v: k for k, v in enumerate(categories)}
        df["store"] = df["store"].map(self.store_mapping)

        df = pd.get_dummies(df, columns=["stateholiday"], drop_first=True)

        self.scaler = StandardScaler()
        self.scaler.fit(df[self.num_cols])
        self.final_columns = df.columns

        return self


    def transform(self, df: pd.DataFrame):
        df = df.copy()

        df["stateholiday"] = df["stateholiday"].astype(str)
        df["schoolholiday"] = df["schoolholiday"].astype(int)
        df["store"] = df["store"].map(self.store_mapping).fillna(-1)
        df = pd.get_dummies(df, columns=["stateholiday"], drop_first=True)

        available_cols = [col for col in self.num_cols if col in df.columns]
        df[available_cols] = self.scaler.transform(df[available_cols])
        
        df = df.reindex(columns=self.final_columns, fill_value=0)

        return df


    