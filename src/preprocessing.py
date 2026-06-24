"""
preprocessing.py

This module loads and preprocesses the housing dataset.
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_dataset():
    """
    Load the California Housing dataset and return it as a DataFrame.
    """
    housing = fetch_california_housing(as_frame=True)
    return housing.frame


def inspect_dataset(df):
    """
    Display basic information about the dataset.
    """
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst Five Rows:")
    print(df.head())


def preprocess_data(df):
    """
    Separate features and target variable.
    """
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    return X, y