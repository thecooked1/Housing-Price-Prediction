import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import json
import os


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Train a Linear Regression model on the provided training data.

    Parameters:
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target values (Price).

    Returns:
    ----------
    LinearRegression
        The fitted Linear Regression model.

    Raises:
    ----------
    ValueError
        If the input data contains null values or incompatible dimensions.
    """
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training Linear Regression: {e}")
        raise


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series, max_depth: int = 5) -> DecisionTreeRegressor:
    """
    Train a Decision Tree Regressor model on the provided training data.

    Parameters:
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target values (Price).
    max_depth : int, optional
        The maximum depth of the tree (default is 5).

    Returns:
    ----------
    DecisionTreeRegressor
        The fitted Decision Tree model.

    Raises:
    ----------
    ValueError
        If the input data is invalid.
    """
    try:
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training Decision Tree: {e})")
        raise


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a regression model using R-squared and Mean Squared Error.

    Parameters:
    ----------
    model : fitted estimator
        The trained ML model to evaluate.
    X_test : pd.DataFrame
        Testing features.
    y_test : pd.Series
        Testing target values.

    Returns:
    ----------
    dict
        A dictionary containing R2 and MSE scores.

    Raises:
    ----------
    Exception
        If evaluation fails.
    """
    try:
        r2 = r2_score(y_test, model.predict(X_test))
        mse = mean_squared_error(y_test, model.predict(X_test))
        return {
            'R2': round(r2, 4),
            'MSE': round(mse, 4)
        }
    except Exception as e:
        print(f"Error evaluating model: {e})")
        raise


def save_results(metrics: dict, filename: str):
    """
    Save model evaluation metrics to a JSON file in the reports/results directory.

    Parameters:
    ----------
    metrics : dict
        Dictionary containing model metrics.
    filename : str
        The name of the file to save (e.g., 'metrics.json').

    Returns:
    ----------
    None
        Writes data to disk.

    Raises:
    ----------
    IOError
        If the directory is not writable.
    """
    try:
        # Determine path based on current directory (notebooks vs root)
        if os.path.basename(os.getcwd()) == 'notebooks':
            output_dir = "../reports/results/"
        else:
            output_dir = "reports/results"

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)

        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved successfully to {file_path}")

    except Exception as e:
        print(f"Error saving results: {e}")
        raise


