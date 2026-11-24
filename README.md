# Customer Churn Analysis and Prediction

A Data Analysis and Machine Learning project developed as part of the Data Analysis Internship at Saiket Systems.

## Project Overview

This project analyzes customer churn in a telecommunications company to identify customer behavior patterns, understand factors associated with churn, segment customers, and predict potential customer churn using Machine Learning.

An interactive Streamlit dashboard is also developed to present analytical insights, model performance, and individual customer predictions.

## Project Tasks

### 1. Data Cleaning and Preprocessing
- Handled missing values.
- Converted categorical variables into numerical representations.
- Applied binary encoding and one-hot encoding.

### 2. Exploratory Data Analysis (EDA)
- Analyzed overall customer churn rate.
- Explored customer demographics and service usage.
- Investigated relationships between churn, tenure, contract types, and payment methods.
- Created visualizations to communicate key findings.

### 3. Customer Segmentation
- Segmented customers based on tenure, monthly charges, and contract type.
- Compared churn rates across different customer segments.
- Identified high-risk customer groups.

### 4. Churn Prediction
- Developed a Logistic Regression classification model.
- Used feature scaling and class balancing.
- Evaluated the model using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Saved the trained model for dashboard integration.

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 74.02% |
| Precision | 50.69% |
| Recall | 78.61% |
| F1-Score | 61.64% |
| ROC-AUC | 84.13% |

## Interactive Dashboard

The Streamlit dashboard includes:

- **Overview:** Key business metrics and overall churn insights.
- **Exploratory Analysis:** Interactive charts and customer behavior analysis.
- **Customer Segmentation:** Customer groups and their churn rates.
- **Model Performance:** Evaluation metrics and confusion matrix.
- **Churn Prediction:** Individual customer churn prediction, probability, and recommended actions.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Plotly
- Streamlit
- Jupyter Notebook
- Joblib

## Project Structure

```text
Customer-Churn-Analysis/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── Dataset (not included)
│
├── models/
│   ├── logistic_regression_model.pkl
│   └── feature_names.json
│
├── notebooks/
│   ├── 01_Data_Cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Customer_Segmentation.ipynb
│   └── 04_Churn_Prediction.ipynb
│
└── visualizations/
    └── Analysis charts
```

## How to Run the Dashboard Locally

### 1. Clone the Repository

```bash
git clone https://github.com/BasmaMohamed123/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
```

### 2. Create a Python Environment

```bash
conda create -n saiket python=3.11
conda activate saiket
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Create a folder named `data` in the project directory and place the original dataset inside it with this exact filename:

`Telco_Customer_Churn_Dataset .csv`

The dataset is not included in this repository.

### 5. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at:

`http://localhost:8501`

## Notes

- The trained model and feature configuration are included in the repository.
- The original dataset must be obtained separately.
- Model performance is based on the project's test dataset and may vary with other data.

## Author

**Basma Mohamed Saber**

Data Analysis | Machine Learning | Python