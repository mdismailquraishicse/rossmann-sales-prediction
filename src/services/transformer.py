import pandas as pd
from sklearn.preprocessing import StandardScaler




class FeatureTransformer:


    def __init__(self):

        self.num_cols = ["month", "week_of_year"]
        self.scaler = None
        self.final_columns = None
        self.store_mapping = None


    def fit(self, df: pd.DataFrame):

        df = df.copy()
        if "sales" in df.columns:
            df = df.drop("sales", axis=1)

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
        if "sales" in df.columns:
            df = df.drop("sales", axis=1)

        df["stateholiday"] = df["stateholiday"].astype(str)
        df["schoolholiday"] = df["schoolholiday"].astype(int)
        df["store"] = df["store"].map(self.store_mapping).fillna(-1)
        df = pd.get_dummies(df, columns=["stateholiday"], drop_first=True)

        available_cols = [col for col in self.num_cols if col in df.columns]
        df[available_cols] = self.scaler.transform(df[available_cols])
        
        df = df.reindex(columns=self.final_columns, fill_value=0)

        return df
    