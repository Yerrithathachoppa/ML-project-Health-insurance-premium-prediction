# 🏥 Health Insurance Premium Predictor

A machine learning-powered web application that predicts health insurance premium costs for individuals based on their personal, medical, and lifestyle attributes. Built with **Streamlit** and powered by **XGBoost** regression models, the app provides accurate, real-time premium estimates through an intuitive UI.

---

## 📌 Project Overview

Health insurance premium pricing is influenced by a wide range of factors including age, BMI, medical history, smoking habits, and more. This project addresses the regression problem of predicting the annual health insurance cost (premium) for a given individual.

The solution implements a **segmented modelling strategy**:
- A dedicated model for **young individuals (age ≤ 25)**
- A separate model for the **rest of the population (age > 25)**

This dual-model approach improves prediction accuracy by accounting for the vastly different risk profiles between younger and older policyholders.

---

## 🚀 Live Demo / How to Run

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ML-project-Health-insurance-premium-prediction.git
cd ML-project-Health-insurance-premium-prediction

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # On Windows
# source venv/bin/activate   # On macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run main.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🗂️ Project Structure

```
ML-project-Health-insurance-premium-prediction/
│
├── artifacts/                    # Pre-trained model & scaler files
│   ├── model_young.joblib        # XGBoost model for age ≤ 25
│   ├── model_rest.joblib         # XGBoost model for age > 25
│   ├── scaler_young.joblib       # Feature scaler for young segment
│   └── scaler_rest.joblib        # Feature scaler for rest segment
│
├── main.py                       # Streamlit web application (UI layer)
├── prediction_helper.py          # ML pipeline: preprocessing & prediction logic
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
```

---

## 🧾 Input Features

The application takes the following 12 inputs from the user:

| Feature | Type | Description |
|---|---|---|
| **Age** | Numeric | Age of the individual (18 – 100) |
| **Number of Dependants** | Numeric | Number of dependants (0 – 20) |
| **Income in Lakhs** | Numeric | Annual income in Indian Rupees (Lakhs) |
| **Genetical Risk** | Numeric | Score representing hereditary health risk (0 – 5) |
| **Insurance Plan** | Categorical | Plan tier: Bronze, Silver, or Gold |
| **Employment Status** | Categorical | Salaried, Self-Employed, or Freelancer |
| **Gender** | Categorical | Male or Female |
| **Marital Status** | Categorical | Married or Unmarried |
| **BMI Category** | Categorical | Normal, Obesity, Overweight, or Underweight |
| **Smoking Status** | Categorical | No Smoking, Occasional, or Regular |
| **Region** | Categorical | Northwest, Southeast, Northeast, or Southwest |
| **Medical History** | Categorical | Pre-existing conditions (e.g., Diabetes, Heart Disease, etc.) |

---

## ⚙️ ML Pipeline

### 1. Feature Engineering — Medical History Risk Score

Medical history is converted into a **normalized risk score** using a domain-driven scoring system:

| Condition | Risk Score |
|---|---|
| No Disease | 0 |
| Thyroid | 5 |
| Diabetes | 6 |
| High Blood Pressure | 6 |
| Heart Disease | 8 |
| Diabetes & High Blood Pressure | 12 |
| High Blood Pressure & Heart Disease | 14 |
| Diabetes & Heart Disease | 14 |
| Diabetes & Thyroid | 11 |

The total score is normalized to **[0, 1]** using the theoretical maximum of **14** (Heart Disease + Diabetes/High BP):

```
normalized_risk_score = total_risk_score / 14
```

### 2. Categorical Encoding

Categorical variables are manually **one-hot encoded** into binary features:

- `gender_Male`
- `region_Northwest`, `region_Southeast`, `region_Southwest`
- `marital_status_Unmarried`
- `bmi_category_Obesity`, `bmi_category_Overweight`, `bmi_category_Underweight`
- `smoking_status_Occasional`, `smoking_status_Regular`
- `employment_status_Salaried`, `employment_status_Self-Employed`
- `insurance_plan` → Ordinal encoded: Bronze=1, Silver=2, Gold=3

### 3. Feature Scaling

Continuous features (`age`, `income_lakhs`) are scaled using **segment-specific StandardScalers** loaded from pre-fitted `.joblib` files — one for the young segment and one for the rest.

### 4. Segmented Prediction

```
if age <= 25:
    prediction = model_young.predict(scaled_features)
else:
    prediction = model_rest.predict(scaled_features)
```

The final output is the **predicted annual insurance premium in Indian Rupees (₹)**.

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|---|---|
| **Python 3.9+** | Core programming language |
| **Streamlit 1.54** | Web application framework (UI) |
| **XGBoost 3.2** | Gradient Boosting regression models |
| **scikit-learn 1.5** | Feature scaling (StandardScaler) |
| **pandas 2.2** | Data manipulation & preprocessing |
| **NumPy 1.26** | Numerical operations |
| **joblib 1.5** | Model serialization / deserialization |

---

## 📊 Model Details

| Attribute | Young Model (age ≤ 25) | Rest Model (age > 25) |
|---|---|---|
| **Algorithm** | XGBoost Regressor | XGBoost Regressor |
| **File** | `model_young.joblib` | `model_rest.joblib` |
| **Scaler** | `scaler_young.joblib` | `scaler_rest.joblib` |
| **Task** | Regression (Premium Prediction) | Regression (Premium Prediction) |

The two-model approach is motivated by the fact that younger individuals (≤ 25) have a fundamentally different insurance risk profile than older adults, warranting separate training to avoid one group dominating the model's learning.

---

## 📋 Requirements

```
pandas==2.2.0
numpy==1.26.3
joblib==1.5.3
streamlit==1.54.0
xgboost==3.2.0
scikit-learn==1.5.0
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🔮 Sample Usage

1. Launch the app with `streamlit run main.py`
2. Fill in the input fields:
   - Age: `35`, Dependants: `2`, Income: `12 Lakhs`
   - Genetical Risk: `2`, Insurance Plan: `Silver`
   - Gender: `Male`, BMI: `Overweight`, Smoking: `No Smoking`
   - Region: `Northwest`, Medical History: `Diabetes`
3. Click **Predict**
4. The app displays: `Predicted Health Insurance Cost: ₹XXXXX`

---

## 🧠 Key Design Decisions

- **Dual-model segmentation** by age bracket improves accuracy over a single global model by reducing age-induced distribution skew.
- **Normalized medical risk score** converts complex multi-condition history into a single, interpretable continuous feature while preserving relative severity ordering.
- **Ordinal encoding for insurance plan** (Bronze < Silver < Gold) correctly captures the tier hierarchy rather than treating it as nominal.
- **Streamlit** was chosen for its simplicity and speed in building data science web apps with minimal boilerplate.

---

## 🙏 Acknowledgements

- Project built as part of the **Codebasics Machine Learning Course** ([codebasics.io](https://codebasics.io))
- © Codebasics — All rights reserved

---

## 📄 License

This project is intended for educational purposes. Please refer to the original course material at [codebasics.io](https://codebasics.io) for licensing information.
