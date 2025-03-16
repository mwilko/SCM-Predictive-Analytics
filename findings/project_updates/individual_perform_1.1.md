# Predictive ability on customers can be dependant on selecting the correct model

## FRE Predictive ability:

### Individual model performance
- RF

| Evaluation Metric | Training Performance | Validation Performance |
| ----------------- | -------------------- | ---------------------- |
| MAE               | 5685.5635           | 14788.7204             |
| MSE               | 243236705.4644     | 1986425980.1777       |
| RMSE              | 15596.0478          | 44569.3390            |
| R²                | 0.9861               | 0.9079                 |

- NN

| Evaluation Metric | Training Performance | Validation Performance |
| ----------------- | -------------------- | ---------------------- |
| MAE               | 12886.8276           | 19456.1654             |
| MSE               | 516493041.2293     | 1162101342.4927       |
| RMSE              | 22726.4833          | 34089.6075            |
| R²                | 0.9704               | 0.9461                 |

- XGBoost

| Evaluation Metric | Training Performance | Validation Performance |
| ----------------- | -------------------- | ---------------------- |
| MAE               | 6550.6235           | 13918.1738             |
| MSE               | 92494728.0000     | 1203561088.0000       |
| RMSE              | 9617.4179          | 34692.3780            |
| R²                | 0.9947               | 0.9442                 |

### Testing ensemble models
