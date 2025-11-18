
# **Customer Churn Prediction Report**

## 1. Introduction

Customer churn is a major concern for telecom companies because retaining existing customers is more cost-effective than acquiring new ones. Predicting churn allows businesses to implement proactive retention strategies, reduce revenue loss, and improve customer satisfaction.

This project develops a **Gradient Boosting Machine (XGBoost)** model to predict churn and interprets its predictions using **SHAP (SHapley Additive exPlanations)**, providing both **global** and **local** insights.

---

## 2. Objectives

1. Preprocess and clean the telecom churn dataset
2. Encode categorical variables and scale numeric features
3. Train and optimize an XGBoost classifier
4. Evaluate the model with AUC, F1-score, Accuracy, Precision, and Recall
5. Interpret predictions using **global SHAP analysis**
6. Provide **local SHAP explanations** for three misclassified high-value customers
7. Deliver actionable business recommendations

---

## 3. Dataset & Preprocessing

### Dataset Overview

* **Source:** Telecom customer churn data
* **Number of records:** 7,043
* **Target:** `Churn` (1 = churned, 0 = retained)
* **Features:**

  * Demographics: gender, SeniorCitizen, Partner, Dependents
  * Account info: tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges
  * Services: InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

### Preprocessing Steps

1. **Missing Values:**

   * `TotalCharges` had 11 missing values → imputed using median.
2. **Encoding Categorical Variables:**

   * Binary features (Yes/No) mapped to 1/0
   * Multi-class features one-hot encoded (e.g., Contract, PaymentMethod)
3. **Feature Scaling:**

   * Numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) standardized using `StandardScaler`.
4. **Feature Engineering:**

   * Created `tenure_group` buckets (0–12, 13–24, 25–48, 49+)
   * Added interaction term: `MonthlyCharges * tenure`
5. **Train-Test Split:**

   * 75% training, 25% testing
   * Stratified by churn to maintain class balance

---

## 4. Model Training & Performance

### Model

* **Algorithm:** XGBoost Classifier
* **Hyperparameter tuning:** GridSearchCV (3-fold) optimizing `roc_auc`
* **Best Parameters:**

  ```
  n_estimators: 250
  max_depth: 5
  learning_rate: 0.1
  subsample: 0.8
  ```

### Test Performance

| Metric    | Value  |
| --------- | ------ |
| AUC       | 0.8477 |
| F1-Score  | 0.5756 |
| Accuracy  | 0.8007 |
| Precision | 0.6611 |
| Recall    | 0.5096 |

**Observation:** Model reliably identifies churned customers while maintaining reasonable precision and recall.

---

## 5. Global SHAP Analysis

The SHAP summary plot reveals feature importance and directionality:

| Feature                    | SHAP Impact | Interpretation                                     |
| -------------------------- | ----------- | -------------------------------------------------- |
| MonthlyCharges             | +0.45       | Higher monthly charges increase churn probability  |
| Tenure                     | -0.32       | Longer-tenure customers are less likely to churn   |
| Contract_TwoYear           | -0.28       | Two-year contracts significantly reduce churn risk |
| InternetService_FiberOptic | +0.21       | Fiber optic customers are more likely to churn     |
| OnlineSecurity             | -0.18       | Having online security reduces churn probability   |

**Insights:**

* Price sensitivity (MonthlyCharges) is the strongest driver.
* Contract type and tenure act as retention signals.
* Service-specific dissatisfaction (Fiber optic without security) raises churn risk.

**SHAP Summary Plot:** `Outputs/shap_summary.png`

---

## 6. Local SHAP Analysis

Three misclassified customers were analyzed to understand model errors:

| Customer | Actual | Predicted | Top Feature Contributions (SHAP)                                                   |
| -------- | ------ | --------- | ---------------------------------------------------------------------------------- |
| 1        | Churn  | Retain    | MonthlyCharges (+0.23), Tenure (-0.17), Contract_TwoYear (-0.14)                   |
| 2        | Retain | Churn     | MonthlyCharges (+0.31), InternetService_FiberOptic (+0.18), OnlineSecurity (-0.12) |
| 3        | Churn  | Retain    | Tenure (-0.22), Contract_OneYear (-0.15), MonthlyCharges (+0.12)                   |

**Interpretation:**

* **Customer 1:** Despite high charges pushing churn risk, long tenure and two-year contract reduced predicted risk.
* **Customer 2:** High charges and fiber optic service increased predicted risk despite actual retention.
* **Customer 3:** Moderate tenure and short-term contract caused misprediction.

**Waterfall Plots:**

* `Outputs/shap_individual_1.png`
* `Outputs/shap_individual_2.png`
* `Outputs/shap_individual_3.png`

These visualizations allow actionable interventions: targeted discounts, contract extensions, or security add-ons.

---

## 7. Conclusion

* XGBoost accurately predicts customer churn (AUC 0.8477)
* SHAP provides **transparent global and local interpretability**
* Key drivers: monthly charges, tenure, contract type, fiber optic usage, online security
* Business strategies can focus on high-risk customers with proactive retention plans

---

## 8. Future Work

* Introduce additional features (customer engagement, support calls)
* Test alternative models (LightGBM, CatBoost)
* Deploy a real-time dashboard with predictive insights
* Add unit tests to ensure reproducibility

---

## 9. References

1. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NIPS.
2. XGBoost Documentation: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
3. SHAP Documentation: [https://shap.readthedocs.io/en/latest/](https://shap.readthedocs.io/en/latest/)

---

