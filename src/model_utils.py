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

# n-beat
from darts import TimeSeries
from darts.models import NBEATSModel


import matplotlib.pyplot as plt


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
    elif model_type == NBEATSModel.__name__:  # N-BEAT Model (Time-Series)
        return {
            # Architecture
            # Look-back window (e.g., 12 months)
            'input_chunk_length': [12, 24],
            'output_chunk_length': [6, 12],  # Forecast horizon
            # Stack count (each learns trend/seasonality)
            'num_stacks': [5, 10],
            'num_blocks': [1, 3],  # Blocks per stack (complexity control)
            'num_layers': [2, 4],  # Layers per block (non-linearity depth)
            'layer_widths': [128, 256],  # Neurons per layer
            'dropout': [0.0, 0.1],  # Regularization for dense layers
            # Training
            'optimizer_kwargs': [{'lr': 1e-3}, {'lr': 1e-4}],  # Learning rate
            'batch_size': [32, 64],  # Smaller batches for stability
            'n_epochs': [50, 100],
            'random_state': [42]
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def find_best_hyperparameters(model, parameter_grid, X_train, y_train):
    model_type = model.__class__.__name__

    print(f'Model type: {model_type}')
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        cv=5,
        n_jobs=-1,  # Uses all CPU cores
        verbose=2,
        # ensures function chooses metrics with the lowest MSE
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)

    print(f'{model.__class__.__name__} Best Parameters: {grid_search.best_params_}')

    return grid_search.best_params_
