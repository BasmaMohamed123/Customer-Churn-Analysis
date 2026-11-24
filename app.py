
from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
VIZ_DIR = BASE_DIR / "visualizations"

# --------------------------------------------------
# Professional Theme
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background-color: #111C35;
        min-width: 265px;
        max-width: 265px;
    }

    section[data-testid="stSidebar"] * {
        color: #E8EDF7;
    }

    section[data-testid="stSidebar"] h1 {
        color: #FFFFFF;
        font-size: 1.35rem;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-size: 1rem;
        font-weight: 500;
        padding: 7px 4px;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: 1rem;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #34415D;
    }

    h1, h2, h3 {
        color: #172B4D !important;
        font-weight: 700 !important;
    }

    h1 {
        font-size: 2rem !important;
    }

    h2 {
        font-size: 1.35rem !important;
    }

    h3 {
        font-size: 1.08rem !important;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 12px;
        padding: 19px 20px;
        box-shadow: 0 2px 8px rgba(20, 40, 80, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #172B4D !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E5EAF2;
        border-radius: 10px;
        overflow: hidden;
    }

    hr {
        border-color: #E2E8F0;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }

    .page-description {
        color: #64748B;
        font-size: 1rem;
        margin-top: -8px;
        margin-bottom: 22px;
    }

    .section-label {
        color: #2563EB;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .insight-card {
        background-color: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-left: 4px solid #2563EB;
        border-radius: 9px;
        padding: 15px 18px;
        margin-bottom: 10px;
        color: #334155;
        font-size: 0.93rem;
    }

    .prediction-result {
        background-color: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 12px;
        padding: 22px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Data and Model Loading
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "Telco_Customer_Churn_Dataset .csv")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "logistic_regression_model.pkl")


@st.cache_data
def load_feature_names():
    with open(MODEL_DIR / "feature_names.json", "r") as file:
        return json.load(file)


df = load_data()
model = load_model()
feature_names = load_feature_names()


# --------------------------------------------------
# Chart Formatting
# --------------------------------------------------
def format_chart(fig, height=350):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=35, b=20),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#475569"
        ),
        title_font=dict(
            size=15,
            color="#172B4D"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#E2E8F0"
    )

    fig.update_yaxes(
        gridcolor="#EDF1F6",
        zeroline=False
    )

    return fig


def show_page_header(title, description):
    st.markdown(f"""
        <div class="section-label">Customer Churn Analytics</div>
        <h1>{title}</h1>
        <p class="page-description">{description}</p>
    """, unsafe_allow_html=True)


def get_churn_rate(data, group_column):
    return (
        data.groupby(group_column, observed=False)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
    )


# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
with st.sidebar:
    st.title("Customer Analytics")
    st.caption("Retention & Predictive Insights")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Exploratory Analysis",
            "Customer Segmentation",
            "Model Performance",
            "Churn Prediction"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.caption("PROJECT")
    st.markdown("Telco Customer Churn")
    st.caption("Data Analysis & Machine Learning")


# --------------------------------------------------
# Overview
# --------------------------------------------------
if page == "Overview":

    show_page_header(
        "Customer Churn Overview",
        "A consolidated view of customer retention, churn patterns, and business indicators."
    )

    total_customers = len(df)
    churned = (df["Churn"] == "Yes").sum()
    churn_rate = churned / total_customers * 100
    retained = total_customers - churned

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned:,}")
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("Retained Customers", f"{retained:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Churn Distribution")

        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Customers"]

        fig = px.pie(
            churn_counts,
            names="Churn",
            values="Customers",
            hole=0.62,
            color="Churn",
            color_discrete_map={
                "Yes": "#E45B65",
                "No": "#356AE6"
            }
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hole=0.62
        )

        fig = format_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Churn Rate by Contract")

        contract_data = get_churn_rate(df, "Contract")

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Churn Rate",
            color="Contract",
            color_discrete_sequence=["#356AE6", "#6D91E8", "#A9BDEB"],
            text_auto=".2f"
        )

        fig.update_traces(
            texttemplate="%{text}%",
            textposition="outside",
            cliponaxis=False
        )

        fig.update_layout(showlegend=False)
        fig = format_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Key Business Insights")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.markdown("""
        <div class="insight-card">
            <b>Overall Retention</b><br>
            The overall customer churn rate is 26.54%, indicating
            an opportunity to strengthen retention strategies.
        </div>
        <div class="insight-card">
            <b>Contract Type</b><br>
            Month-to-month customers show the highest churn rate
            compared with longer-term contracts.
        </div>
        """, unsafe_allow_html=True)

    with insight_col2:
        st.markdown("""
        <div class="insight-card">
            <b>Customer Tenure</b><br>
            Customers with shorter tenure are more likely to leave
            than long-term customers.
        </div>
        <div class="insight-card">
            <b>Monthly Charges</b><br>
            Customers who churn have higher average monthly charges
            than customers who stay.
        </div>
        """, unsafe_allow_html=True)


# --------------------------------------------------
# Exploratory Analysis
# --------------------------------------------------
elif page == "Exploratory Analysis":

    show_page_header(
        "Exploratory Data Analysis",
        "Explore customer demographics, service usage, and billing patterns associated with churn."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Gender")

        gender_data = get_churn_rate(df, "gender")

        fig = px.bar(
            gender_data,
            x="gender",
            y="Churn Rate",
            color="gender",
            text_auto=".2f",
            color_discrete_sequence=["#356AE6", "#8BA8EB"]
        )

        fig.update_layout(showlegend=False)
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig = format_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn by Senior Citizen Status")

        senior_data = get_churn_rate(df, "SeniorCitizen")
        senior_data["SeniorCitizen"] = senior_data["SeniorCitizen"].map(
            {0: "Non-Senior", 1: "Senior"}
        )

        fig = px.bar(
            senior_data,
            x="SeniorCitizen",
            y="Churn Rate",
            color="SeniorCitizen",
            text_auto=".2f",
            color_discrete_sequence=["#8BA8EB", "#356AE6"]
        )

        fig.update_layout(showlegend=False)
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig = format_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tenure Distribution by Churn")

        fig = px.box(
            df,
            x="Churn",
            y="tenure",
            color="Churn",
            color_discrete_map={
                "Yes": "#E45B65",
                "No": "#356AE6"
            }
        )

        fig = format_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Charges by Churn")

        fig = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={
                "Yes": "#E45B65",
                "No": "#356AE6"
            }
        )

        fig = format_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Churn by Payment Method")

    payment_data = get_churn_rate(df, "PaymentMethod")
    payment_data = payment_data.sort_values("Churn Rate", ascending=False)

    fig = px.bar(
        payment_data,
        x="PaymentMethod",
        y="Churn Rate",
        color="Churn Rate",
        text_auto=".2f",
        color_continuous_scale=["#B8C9F0", "#356AE6"]
    )

    fig.update_traces(
        texttemplate="%{text}%",
        textposition="outside",
        cliponaxis=False
    )

    fig = format_chart(fig, height=390)
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Customer Segmentation
# --------------------------------------------------
elif page == "Customer Segmentation":

    show_page_header(
        "Customer Segmentation",
        "Identify customer groups with different churn risk levels based on tenure, charges, and contract type."
    )

    df_seg = df.copy()

    df_seg["Tenure Group"] = pd.cut(
        df_seg["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    df_seg["Charges Group"] = pd.cut(
        df_seg["MonthlyCharges"],
        bins=[-1, 40, 70, 120],
        labels=["Low Charges", "Medium Charges", "High Charges"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Tenure Group")

        tenure_data = get_churn_rate(df_seg, "Tenure Group")

        fig = px.bar(
            tenure_data,
            x="Tenure Group",
            y="Churn Rate",
            color="Churn Rate",
            text_auto=".2f",
            color_continuous_scale=["#F5C6C9", "#D94C59"]
        )

        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig = format_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn by Charges Group")

        charges_data = get_churn_rate(df_seg, "Charges Group")

        fig = px.bar(
            charges_data,
            x="Charges Group",
            y="Churn Rate",
            color="Churn Rate",
            text_auto=".2f",
            color_continuous_scale=["#F5C6C9", "#D94C59"]
        )

        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig = format_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Churn by Contract Type")

    contract_data = get_churn_rate(df_seg, "Contract")

    fig = px.bar(
        contract_data,
        x="Contract",
        y="Churn Rate",
        color="Churn Rate",
        text_auto=".2f",
        color_continuous_scale=["#F5C6C9", "#D94C59"]
    )

    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig = format_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("High-Risk Customer Segments")
    st.caption(
        "Segments are ranked by churn rate. Only groups with at least 30 customers are displayed."
    )

    combined = (
        df_seg.groupby(
            ["Tenure Group", "Charges Group", "Contract"],
            observed=False
        )
        .agg(
            Customers=("customerID", "count"),
            Churned=("Churn", lambda x: (x == "Yes").sum())
        )
        .reset_index()
    )

    combined = combined[combined["Customers"] >= 30].copy()

    combined["Churn Rate"] = (
        combined["Churned"] / combined["Customers"] * 100
    )

    combined = combined.sort_values("Churn Rate", ascending=False)

    st.dataframe(
        combined.head(15).style.format({
            "Churn Rate": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# Model Performance
# --------------------------------------------------
elif page == "Model Performance":

    show_page_header(
        "Model Performance",
        "Evaluation results for the Logistic Regression churn classification model."
    )

    metrics = {
        "Accuracy": 0.7402,
        "Precision": 0.5069,
        "Recall": 0.7861,
        "F1-Score": 0.6164,
        "ROC-AUC": 0.8413
    }

    cols = st.columns(5)

    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value * 100:.2f}%")

    st.divider()

    st.subheader("Confusion Matrix")

    confusion_path = VIZ_DIR / "06_confusion_matrix.png"

    if confusion_path.exists():
        left, center, right = st.columns([1, 2, 1])

        with center:
            st.image(
                str(confusion_path),
                caption="Logistic Regression — Test Set",
                width=520
            )
    else:
        st.warning("Confusion matrix image was not found.")

    st.divider()

    st.subheader("ROC Curve")

    roc_path = VIZ_DIR / "08_roc_curve.png"

    if roc_path.exists():
        left, center, right = st.columns([1, 2, 1])

        with center:
            st.image(
                str(roc_path),
                caption="ROC Curve — Logistic Regression",
                width=520
            )
    else:
        st.warning("ROC curve image was not found.")

    st.divider()

    st.subheader("Model Interpretation")

    interpretation_col1, interpretation_col2 = st.columns(2)

    with interpretation_col1:
        st.markdown("""
        **Recall — 78.61%**

        The model identifies a significant proportion of customers
        who actually churn.

        **Precision — 50.69%**

        Some customers flagged as high-risk may not actually churn.
        """)

    with interpretation_col2:
        st.markdown("""
        **ROC-AUC — 84.13%**

        The model demonstrates good ability to distinguish between
        churned and retained customers.

        **Next Step**

        Further optimization and validation are recommended before
        any production use.
        """)


# --------------------------------------------------
# Churn Prediction
# --------------------------------------------------
elif page == "Churn Prediction":

    show_page_header(
        "Customer Churn Prediction",
        "Enter customer details to estimate churn probability using the trained classification model."
    )

    st.info(
        "Complete the customer profile below to generate a churn prediction."
    )

    with st.form("customer_prediction_form"):

        st.subheader("Customer Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])

        with col2:
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])

        with col3:
            partner = st.selectbox("Has Partner", ["No", "Yes"])

        col1, col2 = st.columns(2)

        with col1:
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])

        with col2:
            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=72,
                value=12
            )

        st.divider()
        st.subheader("Service Information")

        col1, col2 = st.columns(2)

        with col1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        with col2:
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

        if phone_service == "No":
            multiple_lines = "No phone service"
            st.caption("Multiple Lines: No phone service")
        else:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes"]
            )

        if internet_service == "No":
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"

            st.caption(
                "Internet-dependent services are automatically set to "
                "'No internet service'."
            )
        else:
            col1, col2 = st.columns(2)

            with col1:
                online_security = st.selectbox(
                    "Online Security",
                    ["No", "Yes"]
                )

                online_backup = st.selectbox(
                    "Online Backup",
                    ["No", "Yes"]
                )

                device_protection = st.selectbox(
                    "Device Protection",
                    ["No", "Yes"]
                )

            with col2:
                tech_support = st.selectbox(
                    "Tech Support",
                    ["No", "Yes"]
                )

                streaming_tv = st.selectbox(
                    "Streaming TV",
                    ["No", "Yes"]
                )

                streaming_movies = st.selectbox(
                    "Streaming Movies",
                    ["No", "Yes"]
                )

        st.divider()
        st.subheader("Contract & Billing")

        col1, col2 = st.columns(2)

        with col1:
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

        with col2:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        col1, col2 = st.columns(2)

        with col1:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0
            )

        with col2:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                value=840.0,
                step=10.0
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True,
            type="primary"
        )

    if submitted:

        customer_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "PaperlessBilling": paperless_billing,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaymentMethod": payment_method
        }])

        # Binary encoding
        binary_columns = [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling"
        ]

        for col in binary_columns:
            customer_data[col] = customer_data[col].map({
                "Yes": 1,
                "No": 0,
                "Male": 1,
                "Female": 0
            })

        # One-hot encoding
        categorical_columns = [
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaymentMethod"
        ]

        customer_encoded = pd.get_dummies(
            customer_data,
            columns=categorical_columns,
            dtype=int
        )

        # Match training feature names and order
        customer_encoded = customer_encoded.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Prediction
        prediction = int(model.predict(customer_encoded)[0])

        churn_probability = float(
            model.predict_proba(customer_encoded)[0][
                list(model.classes_).index(1)
            ]
        )

        probability_percent = churn_probability * 100

        st.divider()
        st.subheader("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Estimated Churn Probability",
                f"{probability_percent:.2f}%"
            )
            st.progress(churn_probability)

        with result_col2:
            if prediction == 1:
                st.error("Churn Predicted")

                st.markdown("**Recommended Retention Action**")

                st.write(
                    "Review the customer's contract, pricing, and service "
                    "experience. Consider a personalized retention offer."
                )

            else:
                st.success("No Churn Predicted")

                st.markdown("**Recommended Action**")

                st.write(
                    "Continue regular customer engagement and monitor "
                    "satisfaction to support long-term retention."
                )

        st.caption(
            "The prediction uses the trained Logistic Regression model "
            "with a 0.50 classification threshold. The estimated probability "
            "is a model output, not a guarantee of customer behavior."
        )
