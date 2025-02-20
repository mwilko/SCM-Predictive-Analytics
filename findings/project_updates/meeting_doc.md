# Demand/Stock Forecasting

## Short update:

Finished looking for best models for the dataset. Random Forest came to be the best performing.

Currently after hyperparm training and adding more independent variables i have raised my R2 - 90% and MAE are around 15k.

Orders are labels for mainly retailers and supermarkets to make their products more enticing.

- Orders can be 300k+.
- Orders can be very versitile and not have any trends to suggest a increase/decrease in order quantity.

### Current dashboard (Mock up)

This dash is a mock up and will hopefully be transfered into Power BI. This application is being used because its one of the main BI tools used comercially.

#### Overall look of model predictions (Predictions look low due to the random influxes of certain orders)

![RF - Coefficient of Determination (90%) low side of errors](../images/custom_fre/fre_dash_perform1.png)

#### Showing all Product Order Quantities

The line chart demonstates how there are some orders which are really high than usual which is making the yearly customer quantity charts go off-balance.

![RF - Coefficient of Determination (90%) low side of errors](../images/custom_fre/fre_overall_perform.png)

#### Each Order Quantity - Specifics

Shows how the model acts with all orders. The model captures most of the orders but unexpected increases/decreases can be seen better here.
![RF - Coefficient of Determination (90%) low side of errors](../images/custom_fre/fre_per_prod_perform.png)

#### 2022 - 2025 Variance in Actual vs Predicted values

![RF - Coefficient of Determination (90%) low side of errors](../images/custom_fre/fre_perform_22-25.png)
