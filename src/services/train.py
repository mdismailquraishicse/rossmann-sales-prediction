from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



class Trainer:
    """
        This  class is responsible for model training and evaluation pipeline.
    """



    def __init__(self):
        self.models = {
            "decision_tree": DecisionTreeRegressor(
                random_state=42,
                max_depth=10),

            "random_forest": RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5)
        }
        self.best_model = None
        self.best_score = float("inf")


    def train(self, X_train, y_train, X_val, y_val):
        results = {}

        for name, model in self.models.items():
            print(f"model: {name}")
            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)

            # Evaluate
            mae = mean_absolute_error(y_val, y_pred)

            results[name] = mae
            print(f"{name} MAE: {mae}")

            # Track best model
            if mae < self.best_score:
                self.best_score = mae
                self.best_model = model

        print(f"\nBest Model: {self.best_model}")
        print(f"Best MAE: {self.best_score}")

        return self.best_model, results