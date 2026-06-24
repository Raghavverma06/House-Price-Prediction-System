import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    "best_model.pkl"
)

# Load trained model
model = joblib.load(MODEL_PATH)


def predict_price(
    median_income,
    house_age,
    avg_rooms,
    avg_bedrooms,
    population,
    avg_occupancy,
    latitude,
    longitude,
):
    """
    Predict House Price

    Parameters
    ----------
    median_income : float
    house_age : float
    avg_rooms : float
    avg_bedrooms : float
    population : float
    avg_occupancy : float
    latitude : float
    longitude : float

    Returns
    -------
    float
        Predicted House Price
    """

    data = np.array(
        [[
            median_income,
            house_age,
            avg_rooms,
            avg_bedrooms,
            population,
            avg_occupancy,
            latitude,
            longitude
        ]]
    )

    prediction = model.predict(data)

    return float(prediction[0])


# Test

if __name__ == "__main__":

    price = predict_price(
        median_income=5.0,
        house_age=25,
        avg_rooms=6,
        avg_bedrooms=1,
        population=1500,
        avg_occupancy=3,
        latitude=34.1,
        longitude=-118.2
    )

    print("\nPredicted Price :", round(price, 3))