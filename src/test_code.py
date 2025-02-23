'''
READ PLEASE >>>>>>>>>>>>>>>>>>>>>>>

This file contains code which was used in the notebooks.
It has been put in here for code readability since other methods offer beter performance.

All code stated here could be reused so it is stored here if needed.
'''

# import pandas as pd

# from model_utils import evaluate_sarima_model

# # handle potential column duplication during merge
# product_sales = product_sales.drop(columns=['PhysicalInv'], errors='ignore')  # Drop the column if it exists
# product_sales = pd.merge(product_sales, products_s[['ProductNumber', 'PhysicalInv']], on='ProductNumber')

# # prepare sales data for SARIMAX model
# merged_data['OrderDate'] = pd.to_datetime(merged_data['OrderDate'])  # Convert 'OrderDate' to datetime format

# # set index for time series modeling
# sales_data = df.set_index(['OrderDate', 'ProductNumber'])
# sales_data = sales_data['OrderQuantity']  # Focus on the 'OrderQuantity' column for the analysis

# # resample the data to a monthly frequency and aggregate the order quantity
# sales_data = sales_data.reset_index(level='ProductNumber')  # Reset the 'ProductNumber' index level
# sales_data = sales_data.resample('M').sum()  # Aggregate by month ('M' for monthly)

# # handle missing data (if any)
# sales_data['OrderQuantity'].fillna(0, inplace=True)  # Fill missing order quantities with 0 (if any)

# # split data into training and testing sets
# train_size = int(len(sales_data) * 0.8)  # 80% for training, 20% for testing
# train, test = sales_data.iloc[:train_size], sales_data.iloc[train_size:]

# evaluate_sarima_model(sales_data, train, test)

# ---------------------------------------------------------------------------------------------------------------

# # NN imports
# from sklearn.model_selection import GridSearchCV, cross_val_score
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from scikeras.wrappers import KerasRegressor

# # NN METRICS ARE NOT ARENT BETTER THAN BASE MODELS, SO ITS COMMENTED OUT

# # neural network model
# def create_nn_model(input_shape):
#     model = Sequential()
#     model.add(Dense(64, input_dim=input_shape, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1, activation='linear'))
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
#     return model

# # KerasRegressor
# input_shape = X_train_scaled.shape[1]
# nn_model = KerasRegressor(build_fn=create_nn_model, input_shape=input_shape, verbose=1)

# grid_search = GridSearchCV(estimator=nn_model, param_grid=param_grids(nn_model.__class__.__name__), cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(X_train_scaled, y_train)

# nn_params = grid_search.best_params_
# print(f'Best Parameters: {nn_params}')

# nn_model.set_params(**nn_params)

# # print scores for each cross validation
# scores = cross_val_score(nn_model, X_train_scaled, y_train, cv=5)
# print(scores)

# # train
# history = nn_model.fit(X_train_scaled, y_train, epochs=nn_params['epochs'], batch_size=nn_params['batch_size'], validation_split=0.2, verbose=1)

# y_pred_nn = nn_model.predict(X_val_scaled)

# evaluate_model(nn_model, X_train_scaled, y_train)


'''
Keras Regressor (NN) for individual:
'''

# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from scikeras.wrappers import KerasRegressor
# from sklearn.model_selection import RandomizedSearchCV

# # check dataset shape
# print("x_train shape:", X_train.shape)
# print("x_val shape:", X_val.shape)

# # convert x_train to dataframe if needed
# if isinstance(X_train, np.ndarray):
#     X_train = pd.DataFrame(X_train)
#     X_val = pd.DataFrame(X_val)

# # separate numerical and categorical columns
# numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
# categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

# # preprocessing pipelines
# num_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())
# ])
# cat_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
# preprocessor = ColumnTransformer([
#     ('num', num_pipeline, numerical_cols),
#     ('cat', cat_pipeline, categorical_cols)
# ])

# # transform data
# X_train_transformed = preprocessor.fit_transform(X_train)
# print(f"Original features: {X_train.shape[1]}")
# print(f"Transformed features: {X_train_transformed.shape[1]}")

# X_val_transformed = preprocessor.transform(X_val)
# X_val_scaled = np.array(X_val_transformed)

# # convert sparse matrices if necessary
# if hasattr(X_train_transformed, "toarray"):
#     X_train_transformed = X_train_transformed.toarray()
#     X_val_transformed = X_val_transformed.toarray()

# # ensure numpy array format
# X_train_scaled = np.array(X_train_transformed)
# X_val_scaled = np.array(X_val_transformed)
# if X_train_scaled.ndim == 1:
#     X_train_scaled = X_train_scaled.reshape(-1, 1)
# if X_val_scaled.ndim == 1:
#     X_val_scaled = X_val_scaled.reshape(-1, 1)

# # define neural network model
# def create_nn_model(input_shape):
#     model = Sequential([
#         Dense(64, input_dim=input_shape, activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(1, activation='linear')
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
#     return model

# input_shape = X_train_scaled.shape[1]

# # set hyperparameter grid
# def param_grids():
#     return {
#         'epochs': [10, 20, 30],
#         'batch_size': [16, 32, 64]
#     }

# # wrap model for sklearn compatibility
# nn_model = KerasRegressor(build_fn=create_nn_model, input_shape=input_shape, verbose=1)

# grid_search = RandomizedSearchCV(
#     estimator=nn_model,
#     param_distributions=param_grids(),
#     n_iter=10,
#     cv=3,
#     n_jobs=-1,
#     verbose=2
# )

# grid_search.fit(X_train_scaled, y_train)

# # train final model with best parameters
# nn_params = grid_search.best_params_
# best_nn_model = create_nn_model(input_shape)
# best_nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# best_nn_model.fit(
#     X_train_scaled, y_train,
#     epochs=nn_params['epochs'],
#     batch_size=nn_params['batch_size'],
#     validation_split=0.2,
#     verbose=1
# )

# # make predictions
# y_pred_nn = best_nn_model.predict(X_val_scaled)

# evaluate_model_nn(best_nn_model, X_val_scaled, y_val)

'''
Voting (RF & NN) Config for individuals:
'''

# from sklearn.ensemble import VotingRegressor

# rf_nn_voting = VotingRegressor(estimators=[('rf', RandomForestRegressor(**rf_params)), ('nn', KerasRegressor(build_fn=create_nn_model, input_shape=input_shape, **nn_params))])
# rf_nn_voting.fit(X_train_scaled, y_train)
# y_pred_rf_nn_voting = rf_nn_voting.predict(X_val_scaled)

# print('---------------------- TRAINING PERFORMANCE ----------------------')
# evaluate_model(rf_nn_voting, X_train_scaled, y_train)
# print('--------------------------------------------')

# print('---------------------- TEST PERFORMANCE ----------------------')
# evaluate_model(rf_nn_voting, X_val_scaled, y_val)
# print('--------------------------------------------')

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Create a DetailedOrderDate column using week and weekday information.
# df['DetailedOrderDate'] = pd.to_datetime(
#     customer_total['order_year'].astype(str) +
#     customer_total['order_week'].astype(str).str.zfill(2) +
#     customer_total['order_weekday'].astype(str),
#     format='%Y%W%w'
# )

# # Subplots
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
# fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust layout for better spacing

# # Bar plot: Actual vs. Predicted Order Quantity
# axes[0, 0].bar(df.loc[y_val.index, 'DetailedOrderDate'], y_val, label='Actual', alpha=0.6)
# axes[0, 0].bar(df.loc[y_val.index, 'DetailedOrderDate'], y_pred_rf_nn_voting, label='Predicted', alpha=0.6)
# axes[0, 0].set_xlabel('Order Date')
# axes[0, 0].set_ylabel('Order Quantity')
# axes[0, 0].set_title(f'{custom_ref} Total Products - Actual vs Predicted (Voting - RF&NN)')
# axes[0, 0].set_ylim(0, 1_000_000)
# axes[0, 0].legend()
# axes[0, 0].tick_params(axis='x', rotation=45)

# # Residual Plot
# residuals = y_val - y_pred_rf_nn_voting
# axes[0, 1].scatter(df.loc[y_val.index, 'DetailedOrderDate'], residuals, alpha=0.6)
# axes[0, 1].axhline(y=0, color='r', linestyle='--')
# axes[0, 1].set_xlabel('Order Date')
# axes[0, 1].set_ylabel('Residuals')
# axes[0, 1].set_title(f'{custom_ref} Total Products - Residual Plot (Voting RF&NN)')
# axes[0, 1].tick_params(axis='x', rotation=45)

# # Pie Chart: Order Quantity per Month (2024)
# order_2024 = df[df['order_year'] == 2024].groupby('order_month')['OrderQuantity'].sum()
# axes[0, 2].pie(order_2024, labels=order_2024.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
# axes[0, 2].set_title(f'{custom_ref} Order Quantity Distribution (2024) - Voting RF&NN')

# # Line Plots: Yearly Trends
# for i, year in enumerate([2022, 2023, 2024]):
#     yearly_df = df[df['order_year'] == year]

#     if yearly_df.empty:
#         continue  # skip if no data for that year

#     # Aggregate OrderQuantity per month
#     monthly_actual = yearly_df.groupby('order_month')['OrderQuantity'].sum()

#     # Convert total_y_pred_rf to Pandas Series with index from customer_total
#     pred_series = pd.Series(y_pred_rf_nn_voting, index=y_val.index)

#     # Group predictions by month
#     monthly_predicted = pred_series.groupby(df.loc[y_val.index, 'order_month']).sum()

#     # Sort for proper plotting
#     monthly_actual = monthly_actual.sort_index()
#     monthly_predicted = monthly_predicted.reindex(monthly_actual.index)  # Ensure same months

#     # Plot actual and predicted values for the year
#     axes[1, i].plot(monthly_actual.index, monthly_actual, label='Actual', alpha=0.6, marker='o', linestyle='-')
#     axes[1, i].plot(monthly_predicted.index, monthly_predicted, label='Predicted', alpha=0.6, marker='x', linestyle='--')

#     axes[1, i].set_xlabel('Month')
#     axes[1, i].set_ylabel('Order Quantity')
#     axes[1, i].set_title(f'{custom_ref} {year} Total Products - Actual vs Predicted (Voting RF&NN)')
#     axes[1, i].set_ylim(0, max(monthly_actual.max(), monthly_predicted.max()) * 1.1)  # Scale Y axis
#     axes[1, i].legend()
#     axes[1, i].tick_params(axis='x', rotation=45)

# # Adjust layout for better spacing between subplots
# plt.tight_layout()
# plt.show()
