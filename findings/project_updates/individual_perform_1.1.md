# Predictive ability on customers can be dependant on selecting the correct model

## FRE Predictive ability:

### Individual model performance
- RF

| Evaluation Metric | Training Performance | Validation Performance |
| ----------------- | -------------------- | ---------------------- |
| MAE               | 5590.5100           | 14522.2595             |
| MSE               | 243812381.7531     | 1965372154.9945       |
| RMSE              | 15614.4927          | 44332.5180            |
| R²                | 0.9860               | 0.9088                 |

- NN

| Evaluation Metric | Training Performance | Validation Performance |
| ----------------- | -------------------- | ---------------------- |
| MAE               | 19078.7063           | 24314.5841             |
| MSE               | 1079621454.2306     | 1829610488.9606       |
| RMSE              | 32857.5936          | 42773.9464            |
| R²                | 0.9382               | 0.9151                 |

- XGBoost

| Evaluation Metric | Training Performance | Validation Performance |
| ----------------- | -------------------- | ---------------------- |
| MAE               | 5408.1704           | 14167.2148             |
| MSE               | 57466368.0000     | 1436993920.0000       |
| RMSE              | 7580.6575          | 37907.7026            |
| R²                | 0.9967               | 0.9334                 |

### Testing ensemble models


