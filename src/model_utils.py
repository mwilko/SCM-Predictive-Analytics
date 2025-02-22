from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
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

# sarimax (times series)
import itertools

import matplotlib.pyplot as plt


def evaluate_model(model, X, y):
    print('Evaluating model...')

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

    # R-squared (Coefficient of Determination)
    r2 = r2_score(y, y_pred)
    print(f"R-squared (R²): {r2:.4f}")


def evaluate_model_nn(model, X, y):
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

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}


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


# evaluate SARIMAX model using grid search and cv
def evaluate_sarimax(params, train_data, test_data):
    """Evaluate SARIMAX model with given parameters and return RMSE."""
    try:
        p, d, q, P, D, Q, s = params  # ensure correct unpacking
        model = SARIMAX(train_data, order=(p, d, q),
                        seasonal_order=(P, D, Q, s))
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=len(test_data))
        predicted_values = forecast.predicted_mean

        rmse = np.sqrt(mean_squared_error(test_data, predicted_values))
        return rmse
    except Exception as e:
        print(f"Error for params {params}: {e}")
        return float('inf')  # output a high RMSE to ignore bad models


def model_metrics(actual, predicted):
    print("\nRunning model_metrics...")

    actual = np.array(actual)
    predicted = np.array(predicted)

    # deal with zero values in actual to avoid division by zero in MAPE
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) /
                   actual[mask])) * 100 if np.any(mask) else np.nan

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAE': mean_absolute_error(actual, predicted),
        'MAPE (%)': mape,
        'R² Score': r2_score(actual, predicted),
    }

    print("\n------------------------------\nSARIMAX Model Evaluation:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")
    print("------------------------------")

    return metrics


def evaluate_sarima_model(sales_dataset, train, test):
    """
    Evaluates SARIMA model's performance.
    """
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    P_values = [0, 1, 2]
    D_values = [0, 1]
    Q_values = [0, 1, 2]
    s_values = [12]

    param_grid = list(itertools.product(p_values, d_values,
                      q_values, P_values, D_values, Q_values, s_values))

    best_rmse = float('inf')
    best_params = None

    for params in param_grid:
        rmse = evaluate_sarimax(
            params, train['OrderQuantity'], test['OrderQuantity'])
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    if best_params is None:
        print("No valid parameters found.")
        return

    print(f'Best Hyperparameters: {best_params}')
    print(f'Best RMSE: {best_rmse}')

    # `best_params` has exactly 7 values (for training)
    if len(best_params) != 7:
        print("Error: `best_params` does not contain 7 elements.")
        return

    p, d, q, P, D, Q, s = best_params

    # fit SARIMAX
    sarima_model = SARIMAX(train['OrderQuantity'], order=(
        p, d, q), seasonal_order=(P, D, Q, s))
    sarima_results = sarima_model.fit()

    # forecast
    forecast_steps = len(test)
    forecast = sarima_results.get_forecast(steps=forecast_steps)
    predicted_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    # eval predictions
    evaluation_results = model_metrics(test['OrderQuantity'], predicted_values)

    # statement to ensure evaluation_results is printed
    print("\nFinal Evaluation Results:", evaluation_results)

    # model summary
    print(sarima_results.summary())

    # forecast
    plt.figure(figsize=(10, 6))
    plt.plot(sales_dataset.index[-len(test):],
             test['OrderQuantity'], label='Actual Sales')
    plt.plot(sales_dataset.index[-len(test):], predicted_values,
             label='Predicted Sales', linestyle='--')
    plt.fill_between(sales_dataset.index[-len(test):], confidence_intervals.iloc[:,
                     0], confidence_intervals.iloc[:, 1], color='gray', alpha=0.2)
    plt.title('SARIMAX Forecast vs Actual Sales')
    plt.legend()
    plt.show()
