# Customer Churn Prediction Project

**Project by:** Ahsrav  

## Overview  
This project predicts whether a customer will **churn** (leave a service) using transactional and demographic data.  

We use a **Gradient Boosting Machine (XGBoost)** for prediction and **SHAP (SHapley Additive exPlanations)** for model interpretability:  
- **Global interpretation**: Identify which features drive churn predictions across all customers.  
- **Local interpretation**: Explain why specific customers were predicted to churn.  

This approach provides actionable business insights for retention strategies.

---

## 📁 Project Structure

churn_project/
├── data/
│ └── telecom_churn.csv
├── src/
│ ├── preprocess.py
│ ├── train_model.py
│ ├── evaluate_model.py
│ ├── shap_analysis.py
│ └── utils.py
├── notebooks/
│ └── churn_project.ipynb
├── outputs/
│ ├── metrics.csv
│ ├── confusion_matrix.png
│ └── shap_plots/
│ ├── customer_1.png
│ ├── customer_2.png
│ └── customer_3.png
├── reports/
│ └── executive_summary.txt
├── requirements.txt
└── README.md


---

## 🔧 Installation & Setup

1. **Clone the repository**  
```bash
git clone https://github.com/<your-username>/churn_project.git
cd churn_project

2. Create a virtual environment

python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows

3. Install Dependencies

pip install -r requirements.txt

4. Prepare dataset

Place telecom_churn.csv in data/ folder.

Ensure all required features are present.

5. Run Colab Notebook

Open colab/churn_project.ipynb

Run all cells sequentially.


📊 Outputs

outputs/metrics.csv — Model evaluation metrics (AUC, F1, Accuracy, Precision, Recall)

outputs/confusion_matrix.png — Confusion matrix plot

outputs/shap_plots/ — SHAP summary and local plots for selected customers.


🔍 Insights from SHAP

Global feature importance: Top 5 drivers of churn

Tenure

MonthlyCharges

Contract type

PaymentMethod

TotalCharges

Local SHAP analysis: Explains why 3 high-value customers were misclassified, giving actionable insights to reduce churn


🧪 Dependencies

pandas, numpy

scikit-learn

xgboost

shap

matplotlib, seaborn

joblib


✨ Future Work

Advanced feature engineering (recency, frequency, tenure segmentation)

Test other ML models (LightGBM, Random Forest)

Hyperparameter optimization with Optuna

Deploy a dashboard/webapp for business insights

Add unit tests for pipeline components



🙏 Acknowledgements

Thanks to XGBoost and SHAP libraries

Inspired by best practices from GitHub data science templates


📄 License

MIT License

Last updated: 2025-11-18