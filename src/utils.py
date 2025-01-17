import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


NUMERICAL_COLUMNS = [
    "age", "income", "employment_length", "loan_amount",
    "loan_interest_rate", "loan_percent_income", "credit_history_length"
]

CATEGORICAL_COLUMNS = [
    "home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"
]


def load_data(filename):
    """
    This function loads the data from the csv file
    and convert it into a pandas dataframe.

    Parameters
    -----------
    filename : string
        location of the csv file

    Returns
    -------
    dataframe : DataFrame
        loaded data in pandas dataframe format
    """
    dataframe = pd.read_csv(filename)
    print(f"Data Shape: {dataframe.shape}")
    return dataframe


def split_input_output(data, target_col):
    """
    This function splits the data into input
    and output based on the target column.

    Parameters
    ----------
    data: DataFrame
        data to be split

    target_col: str
        name of the target column

    Returns
    --------
    X : DataFrame
        feature of dataset,

    y : Dataframe
        column target of dataset
    """
    y = data[target_col]
    X = data.drop(target_col, axis=1)
    print(f"Original data shape: {data.shape}")
    print(f"X data shape: {X.shape}")
    print(f"y data shape: {y.shape}")
    return X, y


def split_train_test(X, y, test_size, random_state=None):
    """
    This function splits the data into train and test sets.

    Parameters
    ----------
    X: DataFrame
        feature of dataset to split

    y: Dataframe
        output of dataset to split

    test_size: double (0 <= test_size <= 1)
        proportion of the split dataset

    random_state: int
        Controls the shuffling applied to the data before applying the split

    Returns
    -------
    split version of the X and y in form X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"X train shape: {X_train.shape}")
    print(f"X test shape: {X_test.shape}")
    print(f"y train shape: {y_train.shape}")
    print(f"y test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def serialize_data(data, path):
    """
    This function serializes the data into a pickle file.

    Parameters
    ----------
    data: DataFrame
        data to be serialized

    path: string
        location of the pickle file

    Returns
    -------
        None
    """
    joblib.dump(data, path)


def deserialize_data(path):
    """
    This function deserializes the data into a pickle file.

    Parameters
    ----------
    path: string
        location of the pickle file

    Returns
    -------
    data : DataFrame
        data to be deserialized
    """
    data = joblib.load(path)
    return data


def concat_data(numerical_data, categorical_data):
    return pd.concat([numerical_data, categorical_data], axis=1)
