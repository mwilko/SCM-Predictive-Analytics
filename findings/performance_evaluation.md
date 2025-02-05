# Model performance evaluation

Throughout the development the model performance is documented as different training additions / validation tests are made.

Random Forest with CV & Outliter removal
| Evaluation Metric | Model |  
| --- | --- |
| MAE | 84860.9381 |
| MSE | 41863556821.9018 |
| RMSE | 204605.8573 |
| R2 | 0.1196 |

Random Forest with CV & Param grid (With outliers)
| Evaluation Metric | Model |  
| --- | --- |
| MAE | 81106.3853 |
| MSE | 29933938412.4621 |
| RMSE | 173014.2723 |
| R2 | 0.2059 |

Random Forest - CV & Outlier removal & Param grid (Doesnt look correct on plot)
| Evaluation Metric | Model |
| --- | --- |
| MAE | 7.2066 |
| MSE | 758.3109 |
| RMSE | 27.5374 |
| R2 | -0.1655 |

### Adding more independant features due to poor performance

#### Reducing test samples to train model based on each unique customer (to try and get better performance metrics)

_including CV & param grid as previous_

Random Forest (terrible performance)
(independants added: order_year, order_month, order_week, order_day)
| Evaluation Metric | Model |
| --- | --- |
| MAE | 217596.9560 |
| MSE | 127122041934.9724 |
| RMSE | 356541.7815 |
| R2 | -0.7587 |

Random Forest (improved from previous)
(independants added: order_year, order_month, order_week, order_day, **PhysicalInv**)
| Evaluation Metric | Model |
| --- | --- |
| MAE | 115681.3014 |
| MSE | 78442116152.2402 |
| RMSE | 280075.1973 |
| R2 | -0.0853 |

#### Notes

Implement feature engineering to see if perfromance could be improved
Maybe change standard scaler? could be effecting weighting?

Seperate products by customer (may help with regression?) (✅)
