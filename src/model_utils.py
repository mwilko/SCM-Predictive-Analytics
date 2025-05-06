from darts.metrics import mae, mse, rmse, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# preprocessing imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

# nn imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor

from tensorflow import keras
from sklearn.base import BaseEstimator, RegressorMixin

# new nn imports
from sklearn.neural_network import MLPRegressor

# xgboost imports
from xgboost import XGBRegressor
# sarimax (times series)
import itertools

# catboost
from catboost import CatBoostRegressor

# n-beat (time series)
from darts import TimeSeries
from darts.models import NBEATSModel
from itertools import product

import matplotlib.pyplot as plt

# TabNet imports
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.base import BaseEstimator, RegressorMixin

def evaluate_model_advanced(model, X, y, y_scaler):  # nn configuration
    """
    Simplified model eval function with description for future reference

    Parameters:
    - model: Trained scikit-learn pipeline
    - X: Features (DataFrame)
    - y: Target values (1D array)
    - y_scaler: Fitted StandardScaler for inverse scaling the target variable
    """
    # Transform X using the preprocessing pipeline
    X_transformed = model.named_steps['preprocessor'].transform(X)

    # Predict on transformed features
    predictions = model.named_steps['mlp'].predict(X_transformed)

    # Reverse target scaling
    y_actual = y_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    y_pred = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    return {
        'MAE': mean_absolute_error(y_actual, y_pred),
        'MSE': mean_squared_error(y_actual, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_actual, y_pred)),
        'R²': r2_score(y_actual, y_pred)
    }


def evaluate_model(model, X, y):
    """Evaluates a model using common regression metrics."""
    # make predictions
    y_pred = model.predict(X)

    # compute metrics
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R²": r2}


# def evaluate_timeseries(model, val_series, val_covariates, horizon, target_scaler=None):
#     """
#     Evaluates a time series model on validation data.
#     Made to evaluate different time series models.

#     Args:
#         model: Trained time-series model (e.g., N-BEATS).
#         val_series (list): List of TimeSeries objects (actual values).
#         val_covariates (list): List of TimeSeries objects (features).
#         horizon (int): Number of steps to forecast.
#         target_scaler (Scaler): Optional scaler to inverse transform predictions.
#     """
#     all_y_true = []  # List for actual data
#     all_y_pred = []  # List for predicted data

#     for series, covariates in zip(val_series, val_covariates):
#         # Forecast `horizon` steps ahead
#         pred = model.predict(
#             n=horizon,
#             series=series[:-horizon],  # Training portion
#             # Features up to forecast start
#             past_covariates=covariates[:-horizon]
#         )

#         # Inverse scaling (if used)
#         if target_scaler:
#             pred = target_scaler.inverse_transform(pred)
#             series = target_scaler.inverse_transform(series)

#         # Extract values
#         y_true = series[-horizon:].univariate_values()  # Last `horizon` steps
#         y_pred = pred.univariate_values()

#         all_y_true.extend(y_true)
#         all_y_pred.extend(y_pred)

#     # Calculate metrics
#     metrics = {
#         "MAE": mean_absolute_error(all_y_true, all_y_pred),
#         "MSE": mean_squared_error(all_y_true, all_y_pred),
#         "RMSE": np.sqrt(mean_squared_error(all_y_true, all_y_pred)),
#         "R²": r2_score(all_y_true, all_y_pred)
#     }

#     print(f"MAE: {metrics['MAE']:.4f}")
#     print(f"MSE: {metrics['MSE']:.4f}")
#     print(f"RMSE: {metrics['RMSE']:.4f}")
#     print(f"R²: {metrics['R²']:.4f}")

#     return metrics

class TabNetRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_d=16,
        n_a=16,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-3,
        optimizer_params=None,
        mask_type="sparsemax",
        device_name="auto",
        # these next ones will be sent to `.fit()`
        max_epochs=50,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        drop_last=False,
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.optimizer_params = optimizer_params or {"lr": 0.03, "weight_decay": 1e-5}
        self.mask_type = mask_type
        self.device_name = device_name
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.drop_last = drop_last
        self._model = None

    def fit(self, X, y):
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self._model = TabNetRegressor(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=self.optimizer_params,
            mask_type=self.mask_type,
            device_name=self.device_name,
        )

        self._model.fit(
            X, y,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            drop_last=self.drop_last,
        )
        return self

    def predict(self, X):
        return self._model.predict(X)

def param_grids(model_type):
    if model_type == RandomForestRegressor.__name__:  # Random Forest Regressor
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['squared_error'],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [42]
        }
    elif model_type == DecisionTreeRegressor.__name__:  # Decision Tree Regressor
        return {
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['squared_error'],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == LinearRegression.__name__:  # Linear Regression
        return {
            'fit_intercept': [True, False],
            'n_jobs': [100, 200, 300],
            'copy_X': [True, False],
            'positive': [True, False]
        }
    # used keras regressor for NN model (old implementation)
    elif model_type == KerasRegressor.__name__:  # NN Model (best params found)
        return {
            'batch_size': [64],
            'epochs': [200],
            'optimizer': ['adam'],
            'loss': ['mean_squared_error'],
            'verbose': [1],
            'random_state': [42],
            'shuffle': [True]
        }
    elif model_type == MLPRegressor.__name__:  # MLPRegressor (Neural Network)
        return {
            # Number of neurons per layer
            'hidden_layer_sizes': [256, 128, 64],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001],  # L2 regularization (weight decay)
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.005, 0.01],
            'max_iter': [500, 1000, 2000],
            'early_stopping': [True],
            'random_state': [42]
        }
    elif model_type == XGBRegressor.__name__:  # XGBoost Regressor
        return {
            'n_estimators': [100, 500],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 8],
            # Can add 0.7 for bigger datasets (removed due to training times)
            'subsample': [0.6, 1.0],
            # Can add 0.7 for bigger datasets (removed due to training times)
            'colsample_bytree': [0.6, 1.0],
            # Minimum loss reduction for further partitioning
            # 'gamma': [0, 0.1, 0.2], # Only include if seeing overfitting
            # L1 regularization (feature selection)
            'reg_alpha': [0.1],
            # L2 regularization (prevents overfitting)
            'reg_lambda': [1],
            'random_state': [42]
        }
    # CatBoost (Gradient-boosting algorithm)
    elif model_type == CatBoostRegressor.__name__:
        return {
            # Number of trees (keep early_stopping_rounds=50)
            'iterations': [500, 1000],
            'learning_rate': [0.03, 0.1],
            'depth': [6, 8],  # Tree depth (6-8 for balance)
            'l2_leaf_reg': [1, 3],  # L2 regularization to prevent overfit
            'subsample': [0.8, 1.0],  # Fraction of data to sample per tree
            # Fraction of features to use per level
            'colsample_bylevel': [0.8, 1.0],
            'min_data_in_leaf': [1, 5],  # Avoid overfitting to small leaves
            'grow_policy': ['SymmetricTree', 'Depthwise'],
            'random_state': [42]
        }
    elif model_type == TabNetRegressorWrapper.__name__:
        return {
            # constructor args
            "n_d": [8, 16],
            "n_a": [8, 16],
            "n_steps": [3, 5],
            "gamma": [1.3], # default
            "lambda_sparse": [1e-3], # default
            "optimizer_params": [{"lr": 0.03, "weight_decay":1e-5}],
            "mask_type": ["sparsemax"],
            # fit() args
            "max_epochs": [20, 50],
            "patience": [5, 10],
            "batch_size": [1024, 2048],
            "virtual_batch_size": [128, 256],
            "drop_last": [False],
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def find_best_hyperparameters(model_class, parameter_grid, X_train, y_train):
    model_type = model_class.__name__

    # Modified due to Time series models, e.g N-BEATS, couldn't use Gridsearch
    if model_type in [
        "RandomForestRegressor", 
        "DecisionTreeRegressor", 
        "LinearRegression", 
        "MLPRegressor", 
        "XGBRegressor", 
        "CatBoostRegressor",
        "TabNetRegressorWrapper" # TabNet model made for GridSearch
        ]:
        # For Scikit-learn models, use GridSearchCV
        print(f"Performing GridSearchCV for {model_type}...")
        grid_search = GridSearchCV(
            estimator=model_class(),
            param_grid=parameter_grid,
            cv=5,
            n_jobs=-1, # Uses all CPU cores
            verbose=2,
            scoring='neg_mean_squared_error'
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f'{model_type} Best Parameters: {best_params}')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return best_params
