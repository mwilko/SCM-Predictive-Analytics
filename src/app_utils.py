import matplotlib.dates as mdates  # Import for date formatting
import numpy as np
import pandas as pd
# Transformation
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# Base models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
# Stacking ensemble
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
# Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_group_zscore(group):  # Function to compute z-scores for each group
    group['z_score'] = np.abs(stats.zscore(group['OrderQuantity']))
    return group


class Evaluation:
    @classmethod
    # Base metrics (Error and Variance) for regression tasks
    def metrics(cls, actual_data, predicted_data):
        '''
        Error: MAE, MSE, RMSE = Metrics to describe pred errors
        Variance: R2 = Indicates variance captured by model preds
        '''
        # Mean Absolute Error (Avgs abs differences of pred and actual)
        mae = mean_absolute_error(actual_data, predicted_data)
        # Mean Squared Error (Squares errors before avg, panalizes larger errors)
        mse = mean_squared_error(actual_data, predicted_data)
        # Root Mean Squared Error (Sqrt of MSE, puts errors into label quantity which is easier to understand)
        rmse = np.sqrt(mse)
        # RSquared (Coefficient of determination by differs since using nonlinear models. Specifies how much variance the model captures)
        r2 = r2_score(actual_data, predicted_data)

        # Use Markdown formatting for line breaks
        metrics = (
            '##### -- Base Metrics, Predicted against Actual data --  \n'
            f'###### Mean Absolute Error (MAE): {mae:.4f}  \n'
            f'###### Root Mean Squared Error (RMSE): {rmse:.4f}  \n'
            f'###### Coefficient of Determination (R²): {r2:.4f}  \n'
            '---------------------  \n'
        )
        return metrics

    @classmethod
    # Additional Bias and Error metrics for regression tasks
    def advanced_metrics(cls, actual_data, predicted_data):
        residuals = predicted_data - actual_data
        actual_mean = np.mean(actual_data)
        pred_mean = np.mean(predicted_data)
        # Epsilon avoids calculation fail if passed values are 0 (Small enough value to not skew score)
        epsilon = 1e-9
        '''
        Bias: NMB, FB = Preds are over or under predicting
        Error: NME, FGE = Scale of pred errors, too high or too low
        '''
        # Normalized Mean Bias (Shows if predictions are too high or low. E.g 0.25 = 25% too high)
        nmb = np.mean(residuals) / (actual_mean + epsilon)
        # Fractional Bias (Similar to NMB but balances pred and actual. Compared to both pred and actual)
        fb = 2 * np.mean(residuals) / (pred_mean + actual_mean + epsilon)

        # Normalized Mean Error (Shows how big errors are on avg)
        nme = np.mean(np.abs(residuals)) / (actual_mean + epsilon)
        # Fractional Gross Error (Similar to NME, balances pred and actual. Compared to both pred and actual)
        fge = 2 * np.mean(np.abs(residuals)) / \
            (pred_mean + actual_mean + epsilon)

        # Use Markdown formatting for line breaks
        metrics = (
            '##### -- Additional Metrics, Predicted against Actual data --  \n'
            f'###### Normalized Mean Bias/Fractional Bias (NMB/FB): {nmb:.4f}/{fb:.4f}  \n'
            f'###### Normalized Mean Error/Fractional Gross Error (NME/FGE): {nme:.4f}/{fge:.4f}  \n'
            '---------------------  \n'
        )
        return metrics


class Transform:
    @staticmethod
    def compute_zscore(df, threshold=3):
        # Check if df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Group by ProductNumber and apply z-score calculation
        df_grouped = df.groupby('ProductNumber').apply(compute_group_zscore)

        # Filter rows where z-score is above threshold
        df_filtered = df_grouped[df_grouped['z_score'] <= threshold]

        return df_filtered

    @classmethod
    def transform_data(cls, X_train, X_val, X):
        # Handle infinities by replacing them with NaN
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_val.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Define categorical and numeric features
        categorical_features = X_train.select_dtypes(
            include=['object']).columns.tolist()
        numeric_features = [
            col for col in X_train.columns if col not in categorical_features]

        # Define transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine transformers into a preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Fit and transform training data, transform validation data
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_val_preprocessed = preprocessor.transform(X_val)

        return X_train_preprocessed, X_val_preprocessed


class Tuning:
    rf_tuned = RandomForestRegressor(  # Tuned model hyperparams from rf ipynb
        criterion='squared_error',
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
    )

    mlp_tuned = MLPRegressor(  # Tuned model hyperparams from nn ipynb
        activation='relu',
        alpha=0.001,
        early_stopping=True,
        hidden_layer_sizes=64,
        learning_rate='adaptive',
        learning_rate_init=0.01,
        max_iter=2000,
        random_state=42,
        solver='adam'
    )

    xbg_tuned = XGBRegressor(  # Tuned model hyperparams from xgb ipynb
        colsample_bytree=1.0,
        learning_rate=0.2,
        max_depth=3,
        n_estimators=500,
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1,
        subsample=1.0
    )

    # Stacking ensemble with RF MLP and MLP

    # Change 'alpha' value for different weighting
    # Learns how to combine model predictions for best result (prevent overfitting)
    meta_model = Ridge(alpha=1.0)

    stacked_ensemble_tuned = StackingRegressor(
        estimators=[
            ('rf_tuned', rf_tuned),
            ('mlp_tuned', mlp_tuned),
            ('xgb_tuned', xbg_tuned)
        ],
        final_estimator=meta_model
    )


class Plots:
    @classmethod
    def overall(cls, df, y_data, x_data, predict_data, custom_ref):
        # Create subplots (2 rows, 2 columns: one for line plot, one for residual plot)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
        fig.subplots_adjust(hspace=0.5, wspace=0.4)

        # Ensure 'OrderDate' is in datetime format
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])

        # Time Series Comparison
        sns.lineplot(x=df.loc[y_data.index, 'OrderDate'],
                     y=y_data, label='Actual', ax=axes[0, 0])
        sns.lineplot(x=df.loc[y_data.index, 'OrderDate'],
                     y=predict_data, label='Predicted', ax=axes[0, 0])
        axes[0, 0].set_title(f'{custom_ref} - Time Series', fontsize=16)
        axes[0, 0].set_xlabel('Date', fontsize=14)
        axes[0, 0].set_ylabel('Order Quantity', fontsize=14)

        # Set major locator to every month
        axes[0, 0].xaxis.set_major_locator(
            mdates.MonthLocator())  # Show one tick per month
        # '%b' for abbreviated month names, '%y' for the last two digits of the year
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter(
            '%b %y'))  # Show Month Year (e.g., 'Jan 24')

        # Format the x-axis labels and avoid overlap
        plt.setp(axes[0, 0].get_xticklabels(),
                 rotation=45, ha='right', fontsize=12)

        # Residual Plot
        residuals = y_data - predict_data
        sns.scatterplot(x=predict_data, y=residuals, alpha=0.6, ax=axes[0, 1])
        axes[0, 1].axhline(0, color='r', linestyle='--')
        axes[0, 1].set_title(f'{custom_ref} - Residuals', fontsize=16)
        axes[0, 1].set_xlabel('Predicted Values', fontsize=14)
        axes[0, 1].set_ylabel('Scaled Residuals', fontsize=14)

        # Actual vs Predicted Scatter Plot
        min_val = min(y_data.min(), predict_data.min())
        max_val = max(y_data.max(), predict_data.max())
        sns.scatterplot(x=y_data, y=predict_data, alpha=0.6,
                        ax=axes[1, 0], label='Predicted')
        axes[1, 0].plot([min_val, max_val], [min_val, max_val],
                        'r--', linewidth=1)  # Reference line
        axes[1, 0].set_title(f'{custom_ref} - Accuracy', fontsize=16)
        axes[1, 0].set_xlabel('Actual Values', fontsize=14)
        axes[1, 0].set_ylabel('Predicted Values', fontsize=14)
        axes[1, 0].legend(fontsize=12)

        # Monthly Trend Comparison
        monthly_data = df[['order_month']
                          ].loc[df.index.intersection(x_data.index)].copy()
        monthly_data['Actual'] = y_data
        monthly_data['Predicted'] = predict_data

        sns.lineplot(x='order_month', y='Predicted',
                     data=monthly_data, label='Predicted', ax=axes[1, 1])
        sns.lineplot(x='order_month', y='Actual', data=monthly_data,
                     label='Actual', ax=axes[1, 1], color='black', linestyle='--')
        axes[1, 1].set_title(
            f'{custom_ref} - Monthly Trend Comparison (2022-2025)', fontsize=16)
        axes[1, 1].set_xlabel('Month', fontsize=14)
        axes[1, 1].set_ylabel('Order Quantity', fontsize=14)

        # Adjust x-tick labels after all plots are drawn
        plt.setp(axes[0, 0].get_xticklabels(),
                 rotation=45, ha='right', fontsize=12)
        plt.setp(axes[1, 1].get_xticklabels(),
                 rotation=45, ha='right', fontsize=12)

        plt.tight_layout()
        return fig  # Return the figure object
