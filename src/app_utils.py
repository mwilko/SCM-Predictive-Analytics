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
import matplotlib as plt
import seaborn as sns

class Transform:
    @classmethod
    def compute_zscore(cls, group, threshold=3):
        # Only compute z-score if there are at least 2 data points in the group
        if len(group) >= 2:
            group['z_score'] = np.abs(stats.zscore(group['OrderQuantity']))
        else:
            group['z_score'] = 0  # or np.nan if preferred
        return group

    @classmethod
    def transform_data(cls, X_train, X_val, features):
        # Define categorical and numeric features
        categorical_features = ['ProductNumber']
        numeric_features = [col for col in features if col not in categorical_features]

        # Preprocessing pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numeric_transformer, numeric_features)
            ]
        )

        # Fit and transform the training data (fit and transform used so the preprocessor learns the features)
        X_train_preprocessed = preprocessor.fit_transform(X_train)

        # Transform the validation data
        X_val_preprocessed = preprocessor.transform(X_val)

        # Extract feature names from the fitted pipeline
        encoded_cat_features = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)

        # Combine categorical and numeric feature names
        all_feature_names = list(encoded_cat_features) + numeric_features

        # # Convert preprocessed data back to DataFrame with correct feature names and index (Uncode if needed)
        # X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=all_feature_names, index=X_train.index)
        # X_val_preprocessed_df = pd.DataFrame(X_val_preprocessed, columns=all_feature_names, index=X_val.index)

        return X_train_preprocessed, X_val_preprocessed

class Tuning:
    rf_tuned = RandomForestRegressor( # Tuned model hyperparams from rf ipynb
        criterion='squared_error',
        max_depth=None, 
        min_samples_leaf=2, 
        min_samples_split=2, 
        n_estimators=200, 
        random_state=42
    )

    mlp_tuned = MLPRegressor( # Tuned model hyperparams from nn ipynb
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

    xbg_tuned = XGBRegressor( # Tuned model hyperparams from xgb ipynb
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
    meta_model = Ridge(alpha=1.0) # Learns how to combine model predictions for best result (prevent overfitting)

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

        # Time Series Comparison
        sns.lineplot(x=df.loc[y_data.index, 'OrderDate'], y=y_data, label='Actual', ax=axes[0, 0])
        sns.lineplot(x=df.loc[y_data.index, 'OrderDate'], y=predict_data, label='Predicted', ax=axes[0, 0])
        axes[0, 0].set_title(f'{custom_ref} Neural Network - Time Series', fontsize=16)
        axes[0, 0].set_xlabel('Date', fontsize=14)  
        axes[0, 0].set_ylabel('Order Quantity', fontsize=14)

        # Residual Plot
        residuals = y_data - predict_data
        sns.scatterplot(x=predict_data, y=residuals, alpha=0.6, ax=axes[0, 1])
        axes[0, 1].axhline(0, color='r', linestyle='--')
        axes[0, 1].set_title(f'{custom_ref} Neural Network - Residuals', fontsize=16)  
        axes[0, 1].set_xlabel('Predicted Values', fontsize=14)  
        axes[0, 1].set_ylabel('Scaled Residuals', fontsize=14)

        # Actual vs Predicted Scatter Plot
        min_val = min(y_data.min(), predict_data.min())
        max_val = max(y_data.max(), predict_data.max())
        sns.scatterplot(x=y_data, y=predict_data, alpha=0.6, ax=axes[1, 0], label='Predicted')
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)  # Reference line
        axes[1, 0].set_title(f'{custom_ref} Neural Network - Accuracy', fontsize=16)
        axes[1, 0].set_xlabel('Actual Values', fontsize=14)
        axes[1, 0].set_ylabel('Predicted Values', fontsize=14)  
        axes[1, 0].legend(fontsize=12)

        # Monthly Trend Comparison
        monthly_data = df[['order_month']].loc[df.index.intersection(x_data.index)].copy()
        monthly_data['Actual'] = y_data
        monthly_data['Predicted'] = predict_data

        sns.lineplot(x='order_month', y='Predicted', data=monthly_data, label='Predicted', ax=axes[1, 1])
        sns.lineplot(x='order_month', y='Actual', data=monthly_data, label='Actual', ax=axes[1, 1], color='black', linestyle='--')
        axes[1, 1].set_title(f'{custom_ref} Neural Network - Monthly Trend Comparison (2022-2025)', fontsize=16)  
        axes[1, 1].set_xlabel('Month', fontsize=14)
        axes[1, 1].set_ylabel('Order Quantity', fontsize=14)

        # Adjust x-tick labels after all plots are drawn
        plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right', fontsize=12)
        plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right', fontsize=12)

        plt.tight_layout()
        plt.show()
