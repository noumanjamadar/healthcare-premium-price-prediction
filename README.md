# 🏥 Healthcare Premium Price Prediction  

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) 
![Scikit-Learn](https://img.shields.io/badge/ML-LinearRegression%20%7C%20XGBoost-orange)
![VIF](https://img.shields.io/badge/Feature--Selection-VIF-green)
![RMSE](https://img.shields.io/badge/RMSE-Low-brightgreen)
![R² Score](https://img.shields.io/badge/R²-High-blue)

---

## 📌 Table of Contents  
- [Project Overview](#-project-overview)  
- [Problem Statement](#-problem-statement)  
- [Dataset](#-dataset)  
- [Features](#-features)  
- [Project Workflow](#-project-workflow)  
- [Model Export](#-model-export-)  
- [Project Deliverables](#-project-deliverables)  
- [Key Learnings](#-key-learnings)  
- [Results](#-results)  
- [Tools & Technologies Used](#-tools--technologies-used)  
- [Skills Demonstrated](#%E2%80%8D-skills-demonstrated)  
- [Connect with Me](#-connect-with-me)  

---

## 🚀 Project Overview  

This project predicts **health insurance premium prices** using customer demographics, lifestyle habits, and medical risk indicators.

During model development, an insight from **Error Analysis** showed:
> Premium pricing behavior significantly differs by Age Group.

So, a **dual-model strategy** was implemented:
- Linear Regression → Age **≤ 25**
- XGBoost Regressor → Age **> 25**

Multicollinearity was reduced using **Variance Inflation Factor (VIF)** for a more stable and interpretable ML solution.

---

## ❗ Problem Statement  

**Shield Insurance Company** faced challenges in accurate pricing due to:
- Diverse customer age segments
- Medical & genetic risk variations
- Fluctuating healthcare expenditure

🎯 **Goal:**  
Develop an ML model to **predict premium price** using:
- Demographics
- Lifestyle patterns
- Risk scores
- Plan type  

✨ Business Benefits:
- Fair and risk-based pricing  
- Improved underwriting  
- Higher profitability  

---

## 📂 Dataset  

Includes real-world factors such as:
- **Age, BMI, genetic risk**
- **Gender, Region, Marital Status**
- **Smoking & Employment**
- **Insurance Plan Type**
- **Medical conditions**

🎯 **Target Variable:**  
`premium_amount`

---

## 🔑 Features  

| Category | Features |
|---------|----------|
| Demographics | age, gender, region, marital_status |
| Lifestyle | smoking_status, employment_status |
| Health Risk | bmi_category, normalized_risk_score, genetical_risk |
| Policy Details | insurance_plan, income_level |

---

## 🛠 Project Workflow  

### 🔍 1️⃣ Exploratory Data Analysis
- Outlier & distribution study  
- Correlation insights  

### 🧹 2️⃣ Data Preprocessing
- Categorical encoding  
- Missing value handling  

### ⚙️ 3️⃣ Feature Engineering
- Added **genetical_risk**
- Used **normalized_risk_score**
- Removed multicollinearity using **VIF**

### 🔀 4️⃣ Model Strategy (Key Insight)
- Age-based segmentation for better performance

### 🧪 5️⃣ Model Training
- **Linear Regression** — Young Group  
- **XGBoost Regressor** — Adult Group  
- Separate **StandardScaler** for each group

### 📊 6️⃣ Evaluation
- R² Score  
- RMSE  
- Error distribution  

---

## 💾 Model Export 🚀

Saved trained artifacts:

model_young_lr.joblib  
xgb_model_old_gr.joblib  
scaler_young_gr.joblib  
scaler_old_gr.joblib  


---

## 📦 Project Deliverables  

📁 Jupyter Notebooks  
📁 Trained ML Models  
📁 Streamlit App for Prediction  
📁 Visual Analysis  
📁 Documentation (this README)

---

## 🎯 Key Learnings  
- Age-based modeling improves accuracy  
- Multicollinearity reduction = better ML stability  
- Regression models differ by customer segments  
- Insurance pricing domain insights  

---

## 📈 Results  

| Model | Age Group | Best Metrics | Interpretation |
|-------|----------|--------------|---------------|
| **Linear Regression** | ≤ 25 years | High R² • Low RMSE | Premium trend is more linear among young |
| **XGBoost** | > 25 years | Higher R² • Lower RMSE | Captures complex health risk interactions |

---

### 🔹 Model Performance Visualizations

<table>
  <tr>
    <td align="center">
      <img src="visuals/actual_vs_predicted_young.png" width="260"/>
      <br/><b>Actual vs Predicted (Young Group)</b><br/>
      Strong linear fit for younger customers.
    </td>
    <td align="center">
      <img src="visuals/error_distribution_young.png" width="260"/>
      <br/><b>Error Distribution (Young Group)</b><br/>
      Minimal prediction deviation.
    </td>
    <td align="center">
      <img src="visuals/actual_vs_predicted_adult.png" width="260"/>
      <br/><b>Actual vs Predicted (Adult Group)</b><br/>
      XGBoost handles nonlinear risk better.
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="visuals/error_distribution_adult.png" width="260"/>
      <br/><b>Error Distribution (Adult Group)</b><br/>
      Balanced predictions across risk levels.
    </td>
    <td align="center">
      <img src="visuals/streamlit_interface.png" width="260"/>
      <br/><b>Streamlit App UI</b><br/>
      Simple UI for premium forecasting.
    </td>
    <td align="center">
      <b>🚀 Final Outcome</b><br/><br/>
      ✔ Higher accuracy after segmentation<br/>
      ✔ Effective business-driven model<br/>
      ✔ Ready for real-world deployment
    </td>
  </tr>
</table>

---

## 🛠 Tools & Technologies Used  

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Linear Regression  
- XGBoost  
- Streamlit  
- Joblib  
- Matplotlib & Seaborn  

---

## 🧑‍💻 Skills Demonstrated  

- Regression Modeling  
- ML Deployment  
- VIF-based feature selection  
- Production-ready Streamlit UI  
- Insurance data analytics  

---

## 🤝 Connect with Me  

📌 GitHub: https://github.com/noumanjamadar  
💼 LinkedIn: https://www.linkedin.com/in/mohammad-navaman-jamadar/  
🌐 Portfolio: https://codebasics.io/portfolio/Mohammad-Navaman-Jamadar

---
