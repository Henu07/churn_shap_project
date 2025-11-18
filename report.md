# **Customer Churn Prediction Report**

## 1. Introduction

Customer churn is a major challenge for telecom companies because retaining existing customers is more cost-effective than acquiring new ones. Predicting churn allows businesses to implement proactive retention strategies, reduce revenue loss, and improve customer satisfaction.

In this project, we build a **Gradient Boosting Machine (XGBoost)** model to predict churn and use **SHAP (SHapley Additive exPlanations)** to explain predictions. This approach provides both **global insights** into overall churn drivers and **local explanations** for individual high-value customers.

---

## 2. Objectives

1. Preprocess and clean the telecom churn dataset.
2. Encode categorical variables and scale numerical features.
3. Train and optimize an XGBoost classifier.
4. Evaluate model performance using AUC, F1-score, accuracy, precision, and recall.
5. Analyze predictions globally using SHAP.
6. Explain misclassified high-value customers using local SHAP values.
7. Provide actionable business recommendations based on the insights.

---

## 3. Dataset & Preprocessing

### Dataset Overview

* **Source:** Telecom customer churn dataset
* **Number of records:** 7,043
* **Target variable:** `Churn` (1 = churned, 0 = retained)
* **Key features:**

  * Demographics: gender, SeniorCitizen, Partner, Dependents
  * Account info: tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges
  * Services: InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

### Preprocessing Steps

1. **Handling Missing Values:**

   * The `TotalCharges` column had 11 missing entries, which were filled using the median value.

2. **Encoding Categorical Variables:**

   * Binary features (Yes/No) were mapped to 1/0.
   * Multi-class features such as `Contract` and `PaymentMethod` were one-hot encoded.

3. **Scaling Numerical Features:**

   * `tenure`, `MonthlyCharges`, and `TotalCharges` were standardized using `StandardScaler`.

4. **Feature Engineering:**

   * `tenure_group` was created with buckets: 0–12, 13–24, 25–48, 49+ months.
   * Interaction feature: `MonthlyCharges * tenure` was added to capture combined effects.

5. **Train-Test Split:**

   * 75% training, 25% testing
   * Stratified on `Churn` to maintain class balance

---

## 4. Model Training & Performance

### Model

* **Algorithm:** XGBoost Classifier
* **Hyperparameter tuning:** GridSearchCV with 3-fold cross-validation optimizing `roc_auc`
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

**Observation:** The model reliably identifies customers at risk of churn while maintaining reasonable precision and recall.

---

## 5. Global SHAP Analysis

The SHAP summary plot reveals the most important features driving churn and their direction of impact:

| Feature                    | SHAP Impact | Interpretation                                                 |
| -------------------------- | ----------- | -------------------------------------------------------------- |
| MonthlyCharges             | +0.45       | Customers with higher monthly charges are more likely to churn |
| Tenure                     | -0.32       | Longer-tenure customers tend to stay                           |
| Contract_TwoYear           | -0.28       | Two-year contracts reduce churn risk                           |
| InternetService_FiberOptic | +0.21       | Fiber optic service customers are slightly more prone to churn |
| OnlineSecurity             | -0.18       | Having online security reduces churn probability               |

**Insights:**

* Monthly charges are the strongest driver of churn.
* Longer contracts and tenure help retain customers.
* Certain services, like fiber optic internet without security add-ons, increase churn likelihood.

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

* **Customer 1:** Despite high monthly charges increasing churn risk, long tenure and a two-year contract helped the model predict retention.
* **Customer 2:** High charges and fiber optic service increased predicted risk, even though the customer actually stayed.
* **Customer 3:** Moderate tenure and short-term contract caused the model to mispredict retention.

**Waterfall Plots:**

* `Outputs/shap_individual_1.png`
* `Outputs/shap_individual_2.png`
* `Outputs/shap_individual_3.png`

These visualizations allow actionable interventions, such as offering targeted discounts, contract extensions, or promoting online security add-ons.

---

## 7. Actionable Recommendations

1. Offer **discounts** or promotions to new customers with high monthly charges.
2. Encourage **contract extensions** for mid-tenure customers to improve retention.
3. Promote **online security packages** to customers using high-risk services (e.g., fiber optic).
4. Monitor **high-value churn-prone segments** for personalized engagement campaigns.

---

## 8. Conclusion

* XGBoost accurately predicts customer churn (AUC 0.8477).
* SHAP provides transparent global and local interpretability.
* Main drivers of churn: monthly charges, tenure, contract type, fiber optic service, and online security.
* Actionable strategies can reduce churn, improve satisfaction, and retain revenue.

---

## 9. Future Work

* Include additional features such as customer engagement metrics and support calls.
* Test alternative models like LightGBM or CatBoost.
* Deploy a real-time dashboard with predictive insights.
* Add unit tests to ensure reproducibility.

---

## 10. References

1. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NIPS.
2. XGBoost Documentation: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
3. SHAP Documentation: [https://shap.readthedocs.io/en/latest/](https://shap.readthedocs.io/en/latest/)

---

