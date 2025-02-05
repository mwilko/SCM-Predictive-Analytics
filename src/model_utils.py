import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# nn imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor


def evaluate_model(model, X, y):
    print('Evaluating model...')

    # Output type of model
    print(f"Model: {model.__class__.__name__}")

    # K-fold cross validation (k=5)
    y_pred = cross_val_predict(model, X, y, cv=5)

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Mean Squared Error (MSE)
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # R-squared (R²)
    r2 = r2_score(y, y_pred)
    print(f"R-squared (R²): {r2:.4f}")


def param_grids(model_type):
    if model_type == RandomForestRegressor.__name__:  # Random Forest Regressor
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['squared_error'],
            'min_samples_leaf': [1, 2, 4]
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
    # used keras regressor for NN model
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
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def find_best_hyperparameters(model, parameter_grid, X_train, y_train):
    model_type = model.__class__.__name__
    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(
        estimator=model, param_grid=parameter_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f'{model.__class__.__name__} Best Parameters: {grid_search.best_params_}')

    return grid_search.best_params_
