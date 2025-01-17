import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import deserialize_data, concat_data, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def drop_duplicate_data(X, y):
    """
    This function drops duplicated data from row X and y.

    Parameters
    -----------
    X : dataframe
        features of dataset

    y : series
        target of dataset

    Returns
    -------
    X : dataframe
        dropped duplicated data features of dataset

    y : dataframe
        dropped duplicated data target of dataset
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Fungsi median_imputation: parameter X haruslah bertipe DataFrame!")

    if not isinstance(y, pd.Series):
        raise TypeError("Fungsi median_imputation: parameter y haruslah bertipe DataFrame!")

    print(f"Fungis drop_duplicate_data telah divalidasi.")

    X = X.copy()
    y = y.copy()
    print(f"Fungsi drop_duplicate_data: shape dataset sebelum dropping duplicate adalah {X.shape}.")

    X_duplicate = X[X.duplicated()]
    print(f"Fungsi drop_duplicate_data: shape dari data yang duplicate adalah {X_duplicate.shape}.")

    X_clean = (X.shape[0] - X_duplicate.shape[0], X.shape[1])
    print(f"Fungsi drop_duplicate_data: shape dataset setelah drop duplicate seharusnya adalah {X_clean}.")

    X.drop_duplicates(inplace=True)
    y = y[X.index]

    print(f"Fungsi drop_duplicate_data: shape dataset setelah dropping duplicate adalah {X.shape}.")

    return X, y


def numerical_imputer_fit(data):
    numerical_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    numerical_imputer.fit(data)
    return numerical_imputer


def numerical_imputer_transform(data, numerical_imputer):
    imputed_data = numerical_imputer.transform(data)
    imputed_data = pd.DataFrame(imputed_data, columns=data.columns, index=data.index)
    return imputed_data


def ohe_fit(data):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(data)
    return ohe


def ohe_transform(data, ohe):
    encoded_data = ohe.transform(data)
    ohe_data = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(), index=data.index)
    return ohe_data


def scaler_fit(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler


def scaler_transform(data, scaler):
    scaled_data_raw = scaler.transform(data)
    scaled_data_frame = pd.DataFrame(scaled_data_raw, columns=data.columns, index=data.index)
    return scaled_data_frame


def preprocess_data(X_data):
    X_data_numerical = X_data[NUMERICAL_COLUMNS]
    X_data_categorical = X_data[CATEGORICAL_COLUMNS]

    numerical_imputer = deserialize_data("../preprocessing/numerical_imputer.pkl")
    X_numerical_imputed = numerical_imputer_transform(X_data_numerical, numerical_imputer)

    ohe = deserialize_data("../preprocessing/ohe.pkl")
    X_categorical_encoded = ohe_transform(X_data_categorical, ohe)

    X_clean = concat_data(X_numerical_imputed, X_categorical_encoded)

    scaler = deserialize_data("../preprocessing/scaler.pkl")
    X_scaled = scaler_transform(X_clean, scaler)

    return X_scaled
