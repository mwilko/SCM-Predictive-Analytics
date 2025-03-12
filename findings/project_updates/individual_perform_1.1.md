# Predictive ability on customers can be dependant on selecting the correct model

## FRE Predictive ability:
 
- RF
---------------------- TRAINING PERFORMANCE ----------------------
Mean Absolute Error (MAE): 5685.5635
Mean Squared Error (MSE): 243236705.4644
Root Mean Squared Error (RMSE): 15596.0478
R-squared (R²): 0.9861
--------------------------------------------
---------------------- TEST PERFORMANCE ----------------------
Mean Absolute Error (MAE): 14788.7204
Mean Squared Error (MSE): 1986425980.1777
Root Mean Squared Error (RMSE): 44569.3390
R-squared (R²): 0.9079
--------------------------------------------

- NN
---------------------- TRAINING PERFORMANCE ----------------------
Mean Absolute Error (MAE): 12886.8276
Mean Squared Error (MSE): 516493041.2293
Root Mean Squared Error (RMSE): 22726.4833
R-squared (R²): 0.9704
--------------------------------------------
---------------------- TEST PERFORMANCE ----------------------
Mean Absolute Error (MAE): 19456.1654
Mean Squared Error (MSE): 1162101342.4927
Root Mean Squared Error (RMSE): 34089.6075
R-squared (R²): 0.9461
--------------------------------------------

- XGBoost
print('---------------------- TRAINING PERFORMANCE ----------------------')
Mean Absolute Error (MAE): 6550.6235
Mean Squared Error (MSE): 92494728.0000
Root Mean Squared Error (RMSE): 9617.4179
R-squared (R²): 0.9947
--------------------------------------------
print('---------------------- TEST PERFORMANCE ----------------------')
Mean Absolute Error (MAE): 13918.1738
Mean Squared Error (MSE): 1203561088.0000
Root Mean Squared Error (RMSE): 34692.3780
R-squared (R²): 0.9442
--------------------------------------------