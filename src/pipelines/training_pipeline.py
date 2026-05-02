import joblib
import pandas as pd
from src.services.train import Trainer
from sklearn.metrics import mean_absolute_error
from src.services.transformer import FeatureTransformer
from src.services.preprocessing import FeatureEngineering




class TrainingPipeline:

    def __init__(self):

        self.trainer = Trainer()
        self.feature_engineering = FeatureEngineering()
        self.feature_transformer = FeatureTransformer()


    def run_training_pipeline(self):
        
        df_store = pd.read_csv("../data/store.csv")
        df_rossmann = pd.read_csv("../data/Rossmann Stores Data.csv")
        df = self.feature_engineering.merge_and_transform_dfs(df_rossmann=df_rossmann, df_store=df_store, is_train=True)
        df = self.feature_engineering.filter_open_stores(df=df.copy())
        df = self.feature_engineering.data_preprocessing(df=df.copy())

        size = int(df.shape[0]*0.8)
        train = df.iloc[:size]
        test = df.iloc[size:]

        self.feature_transformer.fit(train.copy())
        train_transformed = self.feature_transformer.transform(train)
        test_transformed = self.feature_transformer.transform(test)

        X_train = train_transformed.drop("sales", axis=1)
        y_train = train_transformed["sales"]

        X_test = test_transformed.drop("sales", axis=1)
        y_test = test_transformed["sales"]

        model, result = self.trainer.train(X_train=X_train, y_train=y_train)


        train_prediction = model.predict(X_train)
        test_prediction = model.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_prediction)
        test_mae = mean_absolute_error(y_test, test_prediction)

        print(f"train mae: {train_mae}")
        print(f"test mae : {test_mae}")

        joblib.dump(model, "src/models/random_forest_v1.pkl")
        # model = joblib.load("model.pkl")