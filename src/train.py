"""
train.py

House Price Prediction System
Train multiple machine learning models, compare their performance,
and save the best-performing model.
"""

import os
import joblib

from preprocessing import load_dataset, preprocess_data

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a trained model."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return r2


def main():

    print("\nLoading California Housing Dataset...\n")

    df = load_dataset()

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),
    }

    best_model = None
    best_score = -1
    best_name = ""

    print("\nTraining Models...\n")

    for name, model in models.items():

        model.fit(X_train, y_train)

        score = evaluate_model(
            name,
            model,
            X_test,
            y_test,
        )

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    joblib.dump(
        best_model,
        MODEL_PATH,
        compress=("xz", 3),
    )

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)
    print(f"Model   : {best_name}")
    print(f"R² Score: {best_score:.4f}")

    print("\nModel saved successfully!")
    print(f"Saved at: {MODEL_PATH}")


if __name__ == "__main__":
    main()