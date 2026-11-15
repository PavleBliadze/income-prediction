# Income Prediction

Predicting a bank client's monthly income from transactional and credit data. The project goes from exploratory analysis to a tuned XGBoost model with a custom loss that reduces the bias in the low and high income ranges.

Everything is in one notebook: [`income_prediction.ipynb`](income_prediction.ipynb).

## Data

100,000 clients, 21 numeric features and one target (income, between 200 and 10,000 GEL). The features describe incoming payments, transfers, turnover, balances, liabilities, loans and past income. The data is confidential and is not included in this repository.

Things worth knowing about it:

- every feature is right skewed and has upper outliers;
- most features have some missing values (up to 14% in `Payment_L`);
- several features are strongly correlated with each other, so multicollinearity is an issue for linear models.

## What the notebook does

1. **EDA.** Distributions, outliers by IQR, share of zeros, missing values, Pearson correlation and VIF.
2. **Feature engineering.** Four ratio features, for example salary to monthly payment and past average income to liabilities.
3. **Preprocessing without leakage.** The 80/20 split is done first. KNN imputation (K=5, on standardized data) and IQR capping are fitted on train only and then applied to test. There are two pipelines, one for linear regression and one for tree models.
4. **Models.** OLS and linear regression (with and without a log target), Random Forest and XGBoost, each as a baseline and then tuned to remove overfitting.
5. **Error analysis by income band.** The tuned XGBoost overestimates low incomes and underestimates high ones.
6. **Custom objective.** A segment-aware Huber loss with a threshold penalty that pushes the model to be more careful in the bands where it was biased.
7. **Interpretation.** Split importance, permutation importance on CV folds, partial dependence and SHAP.

## Results on the test set

| Model | R2 | MAPE | MAE |
| --- | ---: | ---: | ---: |
| Linear regression | 0.733 | 0.366 | 637 |
| Linear regression, log target | 0.657 | 0.370 | 713 |
| Random Forest, tuned | 0.746 | 0.375 | 614 |
| XGBoost, tuned | 0.757 | 0.351 | 596 |
| XGBoost with segment-aware loss (final) | 0.712 | 0.346 | 671 |

The final model gives up some R2 on purpose. In income estimation a large overestimate for a low income client is worse than a slightly higher average error, so the last step trades overall fit for a more even error across income bands. The most important features are the client's minimum incoming payment and total incoming payments.

## Reproducibility

The data is confidential and is not part of this repository, so the notebook cannot be re-run from here. All outputs are saved in the notebook, so the full analysis, tables and charts can be read directly on GitHub. `requirements.txt` lists the libraries that were used.
