# Customer Churn Analysis and Prediction

A Data Analysis and Machine Learning project developed as part of the Data Analysis Internship at **Saiket Systems**.

## Project Overview

This project analyzes customer churn in a telecommunications company to identify customer behavior patterns, understand factors associated with churn, segment customers, and predict potential customer churn using Machine Learning.

An interactive Streamlit dashboard was developed to present analytical insights, model performance, individual customer predictions, and recommended retention actions.

## Project Tasks

### 1. Data Cleaning and Preprocessing
- Handled missing values.
- Converted categorical variables into numerical representations.
- Applied binary encoding and one-hot encoding.
- Prepared the dataset for further analysis and machine learning.

### 2. Exploratory Data Analysis (EDA)
- Analyzed the overall customer churn rate.
- Explored customer demographics and service usage.
- Investigated relationships between churn, tenure, contract types, and payment methods.
- Created visualizations to communicate key findings.

### 3. Customer Segmentation
- Segmented customers based on tenure, monthly charges, and contract type.
- Compared churn rates across different customer segments.
- Identified high-risk customer groups.

### 4. Churn Prediction
- Developed a Logistic Regression classification model.
- Applied feature scaling and class balancing.
- Evaluated the model using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Saved the trained model for integration into the dashboard.

## Model Performance

The final Logistic Regression model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 74.02% |
| Precision | 50.69% |
| Recall | 78.61% |
| F1-Score | 61.64% |
| ROC-AUC | 84.13% |

## Dashboard Preview

The project includes an interactive Streamlit dashboard with five main sections:

- **Overview:** Key business metrics and overall churn insights.
- **Exploratory Analysis:** Interactive charts and customer behavior analysis.
- **Customer Segmentation:** Customer groups and their churn rates.
- **Model Performance:** Evaluation metrics, confusion matrix, and ROC curve.
- **Churn Prediction:** Individual customer churn prediction, churn probability, and recommended actions.

### Dashboard Screenshots

| Dashboard Preview | Dashboard Preview |
|---|---|
| ![Screenshot 1](screenshots/Screenshot%202026-11-24%20161812.png) | ![Screenshot 2](screenshots/Screenshot%202026-11-24%20161832.png) |
| ![Screenshot 3](screenshots/Screenshot%202026-11-24%20161931.png) | ![Screenshot 4](screenshots/Screenshot%202026-11-24%20161944.png) |
| ![Screenshot 5](screenshots/Screenshot%202026-11-24%20161957.png) | ![Screenshot 6](screenshots/Screenshot%202026-11-24%20162006.png) |
| ![Screenshot 7](screenshots/Screenshot%202026-11-24%20162027.png) | ![Screenshot 8](screenshots/Screenshot%202026-11-24%20162035.png) |
| ![Screenshot 9](screenshots/Screenshot%202026-11-24%20162054.png) | ![Screenshot 10](screenshots/Screenshot%202026-11-24%20162106.png) |
| ![Screenshot 11](screenshots/Screenshot%202026-11-24%20162115.png) | ![Screenshot 12](screenshots/Screenshot%202026-11-24%20162128.png) |
| ![Screenshot 13](screenshots/Screenshot%202026-11-24%20162136.png) | ![Screenshot 14](screenshots/Screenshot%202026-11-24%20162148.png) |
| ![Screenshot 15](screenshots/Screenshot%202026-11-24%20162158.png) | ![Screenshot 16](screenshots/Screenshot%202026-11-24%20162223.png) |

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Logistic Regression
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit
- **Development Environment:** Jupyter Notebook
- **Model Serialization:** Joblib

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
├── visualizations/
│   ├── 01_overall_churn.png
│   ├── 02_demographics_churn.png
│   ├── 03_tenure_monthly_charges.png
│   ├── 04_contract_payment_churn.png
│   ├── 05_customer_segmentation.png
│   ├── 06_confusion_matrix.png
│   ├── 07_tuned_confusion_matrix.png
│   └── 08_roc_curve.png
│
└── screenshots/
    └── Dashboard screenshots
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

Create a folder named `data` in the project directory and place the original dataset inside it using the following exact filename:

`Telco_Customer_Churn_Dataset .csv`

**Note:** The dataset is not included in this repository and must be obtained separately.

### 5. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at:

`http://localhost:8501`

## Notes

- The trained Logistic Regression model and feature configuration are included in the repository.
- The original dataset must be obtained separately.
- Model performance is based on the project's test dataset.
- The dashboard runs locally using Streamlit.

## Author

**Basma Mohamed Saber**

Data Analysis | Machine Learning | Python