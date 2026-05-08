import joblib
import pandas as pd
from src.services.preprocessing import FeatureEngineering




class PredictionPipeline:




    def __init__(self):
        
        self._required_columns = {'store', 'dayofweek', 'date', 'promo', 'stateholiday', 'schoolholiday'}
        self.model = joblib.load("../src/models/random_forest_v1.pkl")
        self.transformer = joblib.load("../src/models/feature_transformer.pkl")
        self.feature_engineering = FeatureEngineering()


    def run_prediction_pipeline(self, df:pd.DataFrame):
        """
            To predict sales for given data.
            Input:
                df:  merged pandas dataframe of rossmann and stores ['store', 'dayofweek',
                'date', 'promo', 'stateholiday', 'schoolholiday'].
            Output:
                prediction: predicted sales.
        """

        if not self._required_columns.issubset(df.columns):
            print(f"required columns: {list(self._required_columns)}")
            return f"column miss-matched: {list(df.columns)}"

        df.columns = [col.lower().strip() for col in df.columns]
        df = self.feature_engineering.data_preprocessing(df=df.copy())
        df_transformed = self.transformer.transform(df=df.copy())
        prediction = self.model.predict(df_transformed)
        return prediction