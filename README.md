# Amex_Challenge_Offers_Ranking
This repository contains our solution for the American Express Campus Challenge 2025 – Decision Science Track, where the task was to build a machine learning model that ranks personalized offers for customers to maximize engagement. 

**Problem Statement-**
American Express wanted a system that recommends the most relevant offers to each customer, ranked in order of likelihood of interaction (clicks). Given historical customer behavior, offer metadata, and event logs, our goal was to predict the top 7 offers a customer is most likely to engage with – evaluated using Mean Average Precision at 7 (MAP@7).


---

##  Key Components

###  Data Preprocessing

- Loaded and cleaned data from various `.parquet` files.
- Cleaned constant/null-heavy columns and retained categorical dtypes where beneficial.
- Created train and test datasets with consistent schema.

###  Feature Engineering

- Created CTR-based features from events and transactions.
- Offer duration buckets (flash sale, long-term, etc).
- Offer redemption frequency and discount bucket features.

###  Model Training

- Trained using **XGBoost** with **Optuna** for hyperparameter tuning.
- `tree_method='hist'` used for memory-efficient CPU training.
- Final model achieved:
  - **AUC > 0.98**
  - **Average Precision: 0.803**
  - **MAP@7 close to leaderboard benchmark**

###  Evaluation

- Tracked feature importance and validation performance per fold.

---

##  Tech Stack

- Python   
- Pandas, NumPy  
- XGBoost 
- Optuna for hyperparameter optimization  
- Matplotlib & Seaborn for visualization  

---

## 📌 Results & Insights

- Top-ranked features included customer redemption frequency, offer type, and event click counts.
- Combining CTR features from events and offer metadata provided a major lift in performance.
- Effective cleaning by checking correlation with target as well as amongst the columns itself resulted in great dimensionality reduction.

---


