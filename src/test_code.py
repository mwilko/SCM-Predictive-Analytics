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
