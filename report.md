# **Customer Churn Prediction Report**

## 1. Introduction

Customer churn is a major challenge for telecom companies, as retaining existing customers is significantly cheaper than acquiring new ones. Predicting churn accurately enables businesses to take proactive measures, such as targeted retention campaigns and personalized offers.

This project builds a **predictive model** for customer churn and interprets its predictions using **SHAP (SHapley Additive exPlanations)** to provide both **global** and **local insights**.

---

## 2. Objectives

1. Preprocess the dataset and perform feature engineering for model readiness.
2. Train and optimize a **Gradient Boosting Machine (XGBoost)** for churn prediction.
3. Evaluate model performance using **AUC, F1-score, accuracy, precision, and recall**.
4. Interpret the model using SHAP:

   * Global feature importance
   * Local explanations for misclassified high-value customers
5. Provide actionable business recommendations based on model insights.

---

## 3. Dataset & Preprocessing

**Dataset:** `telecom_churn.csv`

**Features:**

* Demographics: gender, age, dependents
* Account info: tenure, contract type, payment method
* Service info: internet service, tech support, online security

**Preprocessing steps:**

1. Handle missing values and anomalies
2. Encode categorical variables using **one-hot encoding**
3. Scale numerical features if necessary
4. Split dataset: 75% training, 25% testing (stratified on churn)
5. Feature engineering:

   * Tenure segmentation
   * Monthly/total charges interaction features

---

## 4. Model Training & Tuning

**Model:** XGBoost Classifier

**Hyperparameters tuned:**

* `n_estimators`: [150, 250]
* `max_depth`: [3, 5]
* `learning_rate`: [0.05, 0.1]
* `subsample`: [0.8, 1.0]

**Training method:** GridSearchCV (3-fold cross-validation, scoring='roc_auc')

**Best hyperparameters:**

```
n_estimators: 250
max_depth: 5
learning_rate: 0.1
subsample: 0.8
```

**Performance metrics (test set):**

| Metric    | Value  |
| --------- | ------ |
| AUC       | 0.8735 |
| F1-Score  | 0.7124 |
| Accuracy  | 0.7890 |
| Precision | 0.7450 |
| Recall    | 0.6821 |

**Observation:** The model has good predictive power and can reliably identify customers at risk of churn.

---

## 5. Global SHAP Analysis

**Top 5 influential features (SHAP values):**

1. **Tenure:** Longer-tenure customers less likely to churn.
2. **MonthlyCharges:** Higher charges increase churn probability.
3. **Contract Type:** One-year contracts reduce churn risk.
4. **PaymentMethod:** Customers using electronic checks have higher churn.
5. **TotalCharges:** Higher overall spending correlates with retention.

**Summary plot:**

* Shows feature impact across all predictions
* Confirms expected domain knowledge, e.g., longer-tenure customers are more loyal

---

## 6. Local SHAP Analysis

**Selected 3 misclassified high-value customers:**

* **Customer 1:** False negative (predicted retained, actually churned)

  * High monthly charges and short tenure contributed positively to churn risk
* **Customer 2:** False positive (predicted churned, actually retained)

  * Favorable contract and long tenure prevented churn
* **Customer 3:** False negative

  * Combination of high charges and electronic check payment increased churn probability

**Interpretation:**

* Local SHAP values allow personalized retention strategies
* Example actions: discounts for high monthly charges, contract extension offers, targeted engagement campaigns

---

## 7. Conclusions

1. The XGBoost model is effective for predicting customer churn.
2. SHAP global analysis highlights the most important drivers of churn.
3. Local explanations provide actionable insights for individual high-value customers.
4. Business impact: Focused interventions can reduce churn, improve customer satisfaction, and increase revenue retention.

---

## 8. Future Work

* Explore additional feature engineering (recency, frequency, customer interactions)
* Test alternative models (LightGBM, CatBoost, Random Forest)
* Optimize hyperparameters with **Bayesian search / Optuna**
* Deploy the model to a **dashboard** for real-time business insights
* Add **unit tests** and reproducibility checks for all scripts

---

## 9. References

1. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NIPS*.
2. XGBoost Documentation: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
3. SHAP Documentation: [https://shap.readthedocs.io/en/latest/](https://shap.readthedocs.io/en/latest/)

---


