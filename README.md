# DA5401 A8: Ensemble Learning for Complex Regression Modeling on Bike Share Data

**Name:** Tanmay Gawande  
**Roll Number:** DA25M030


## Project Overview

This project implements and compares three ensemble learning techniques — **Bagging**, **Boosting**, and **Stacking** — to predict hourly bike rentals using the **Bike Sharing Dataset** from the UCI Machine Learning Repository.  
The objective is to demonstrate how ensemble methods help manage the **bias–variance trade-off** and improve model generalization on a complex regression problem.

---

## Project Structure

```

DA5401-assignment-8/
├── dataset/                                 # Main dataset
│   ├── day.csv
│   ├── hour.csv
│   └── Readme.txt
├── .gitignore                               # Files to be ignored during commits
├── da25m030-assignment-8-solution.ipynb     # Main Jupyter Notebook (solution)
├── DA5401-A8-ensemble-learning.pdf          # Assignment problem statement
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation

````

---

## Dataset Details

**Source:** [UCI ML Repository – Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)  
**Citation:** Fanaee-T, Hadi, and Gamper, H. (2014). *Bike Sharing Data Set*. UCI Machine Learning Repository.  

- **File Used:** `hour.csv`  
- **Samples:** ~17,000 hourly observations  
- **Target Variable:** `cnt` – total count of bike rentals (casual + registered)  
- **Features Used:**
  - **Categorical:** `season`, `mnth`, `hr`, `weekday`, `weathersit`
  - **Binary:** `yr`, `holiday`, `workingday`
  - **Continuous:** `temp`, `atemp`, `hum`, `windspeed`

---

## Methodology

### **Part A: Data Preprocessing and Baseline**
- Dropped irrelevant columns: `instant`, `dteday`, `casual`, `registered`
- One-hot encoded categorical features to prevent false ordinal relationships (e.g., `season 4` ≠ greater than `season 1`)
- Split data chronologically (first 80% train, last 20% test)
- Built **Decision Tree Regressor** (max_depth=6) and **Linear Regression** as baseline models  
  - **Best baseline:** *Linear Regression (RMSE = 116.10)*

### **Part B: Ensemble Models**
- **Bagging (80 Trees)**  
  - Slightly reduced variance vs. single Decision Tree (RMSE: 146.07 → 149.45)  
  - Modest improvement, indicating underfitting of base trees.  
- **Gradient Boosting**  
  - Train RMSE: 63.86, Test RMSE: 86.92  
  - Outperformed both baseline and bagging models, effectively reducing bias.

### **Part C: Stacking**
- **Base Learners (Level-0):**
  - KNN Regressor  
  - Bagging Regressor  
  - Gradient Boosting Regressor  
- **Meta-Learner (Level-1):**
  - Ridge Regression  
- **Result:** RMSE = **81.79** → *Best performing model*  
- Meta-learner gave highest weight to Gradient Boosting, moderate to KNN, and slightly negative to Bagging.

### **Part D: Comparative Analysis**
| Model | Test RMSE |
| :-- | --: |
| **Stacking (KNN + Bag + GBR → Ridge)** | **81.79** |
| Gradient Boosting | 86.92 |
| Linear Regression | 116.10 |
| Bagging (80 Trees) | 146.07 |

---

## Key Insights

- **Bagging** slightly reduced variance but didn’t significantly improve performance — likely due to shallow base trees.  
- **Boosting** effectively reduced bias, achieving a large performance gain over both bagging and baseline models.  
- **Stacking** balanced both bias and variance by leveraging diversity among learners.  
- The **meta-learner (Ridge)** learned to rely more on **Gradient Boosting** (strong learner), while adjusting with **KNN** and down-weighting **Bagging**.

---

## Conclusions

- **Best Model:** Stacking (RMSE = **81.79**)  
- **Why it worked:** It combined the low bias of Boosting with the local adaptability of KNN and the variance reduction from Bagging.  
- **Bias–Variance Summary:**
  - Bagging → ↓ Variance  
  - Boosting → ↓ Bias  
  - Stacking → Optimal trade-off between both  

Overall, stacking produced the most generalizable model for bike rental demand prediction.

---

## Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
````

Main libraries:

* pandas
* numpy
* scikit-learn
* matplotlib
---
Thank you!
---