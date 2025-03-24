# PhantomBuster Churn Prediction Engine 👻

*Unlock the secrets of customer retention with machine learning.* 🔮 

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)

---

## 🎯 Problem Statement  
PhantomBuster’s Executive Committee seeks to proactively identify **at-risk customers** and reduce churn by leveraging data-driven insights. This project delivers a predictive model to flag potential churn risks and uncovers actionable strategies to improve retention.

---

## 🚀 Features  
- **Predictive Power**: Identifies customers likely to churn with 75%+ AUC-ROC.  
- **Actionable Insights**: Highlights key drivers of churn.  
- **Scalable Pipeline**: End-to-end workflow from EDA to deployment-ready models.  
- **Explainability**: Permutation importance for stakeholder transparency.  

---

## 📊 Data & Approach  
### Dataset  
- **Customer Subscriptions**: Plan name, plan age, billing period, currency, monthly payment amount, country code. 
- **Product Usage**: Activity day count, support ticket count, day since last login.  
- **Churn Labels**: Churn status (1/0).  

### Technical Pipeline  
1. **Exploratory Data Analysis (EDA)**  
   - Visualized churn patterns and correlations.  
   - Detected class imbalance (~15/16% churn rate).  
   - Identify how to manage missing values.
2. **Feature Engineering**  
   - Encode categorical features (if necessary). 
   - Normalized quantitative features (if necessary).  
   - Impute missing values (if necessary).
3. **Modeling**  
   - Test several binary classification models.  
   - Test different sets of features (Bruteforce).
   - Optimize hyperparameters with Optuna (Bayesian search).  
   - Calibrate model for better probability estimate.
   - Tune positive label threshold.
4. **Evaluation**  
   - **Best Model**: HistGradientBoosing (AUC-ROC: 0.76).  
   - Key features: `DELINQUENCY_DAY_COUNT`, `PLAN_AGE`, `BILLING_PERIOD`.  

---

## 💡 Business Impact  
- **Retention Strategies**: Target high-risk customers with personalized incentives.  
- **Executive Action**: Monthly churn risk reports + automated alert system.  

---

## 📂 Repository Structure
```
├── README.md                         <- The top-level README for developers using this project.
│
├── TIMELINE.md                       <- The time allocation for the project.
│
├── data
│   ├── dataset.csv                   <- The original, immutable data dump.
│   ├── train.csv                     <- The subsample of the original dataset used to train the model.
│   ├── test.csv                      <- The subsample of the original dataset used to evaluate the final model.
│   ├── pred.csv                      <- The predictions obtained doing inference on the test set using the final model.
│   └── optuna_study.db               <- SQLite databse which store the Optuna metadata concerning hyperparameters tuning.
│
├── models                            <- Trained and serialized models.
│
├── notebooks                         <- Jupyter notebooks. Naming convention is a number (for ordering).
│
├── reports                           <- Generated analysis as HTML, PDF, LaTeX, etc.
│  
├── pyproject.toml                    <- Project configuration file with package metadata for churn_classification_engine and configuration for tools like ruff.
│
├── uv.lock                           <- A cross-platform lockfile that contains exact information about the project's dependencies
│
├── .python-version                   <- Python-version for this project.
│
└── churn_classification_engine       <- Source code.
    ├── __init__.py                   <- Makes churn_classification_engine a Python module.
    ├── config.py                     <- Store useful variables and configuration.
    ├── data
    │   ├── __init__.py
    │   ├── get-data.py               <- Scripts to download data.
    │   └── split-data.py             <- Code to split the data in train and test sets.
    └── model                
        ├── __init__.py 
        ├── hyperparameters-tuning.py <- Code to find the best hyperparameters for selected model.  
        ├── train.py                  <- Code to train & calibrate the final model with the train set.          
        ├── predict-proba.py          <- Code to get the churn probability of customers in test set.            
        └── evaluate.py               <- Code to generate an HTML report with main evaluation metrics.

```

---

## 🛠️ Installation & Usage 
1. Download project 
```bash
git clone https://github.com/paulsteffen-lab/phantombuster-casestudy.git
```

2. Create virtual environment
```bash
uv sync
```

3. Get data
```bash
uv run churn_classification_engine/data/get-data.py
```

4. Split data
```bash
uv run churn_classification_engine/data/split-data.py
```

5. Hyperparameters tuning (optional)
```bash
uv run churn_classification_engine/model/hyperparameters-tuning.py
```

6. Train the classifier pipeline on the train set
```bash
uv run churn_classification_engine/model/train.py
```

6. Predict churn probability on the test set
```bash
uv run churn_classification_engine/model/predict-proba.py
```

7. Evaluate the prediction
```bash
uv run churn_classification_engine/model/evaluate.py
```