import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree


# Evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# For combining pipelines after encoding
from sklearn.compose import make_column_selector as selector

sns.set_theme(style="whitegrid")

# App Title
st.title("Screen Time and Productivity Analysis Dashboard")
st.markdown("An interactive analysis of behavioral patterns and their impact on productivity.")


# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Explore",
                            ["Overview", "EDA", "Modeling & Evaluation", "Insights" ])


# ----------------------
# Interactive EDA Section
# ----------------------

@st.cache_data
def get_data():
    return pd.read_csv("Smartphone_Usage_Productivity_Dataset_50000.csv")

df = get_data()

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include="object").columns.tolist()

@st.cache_data
def get_hist(df, color_col, hist_column):
    fig_hist = px.histogram(df, x=hist_column, color=color_col, marginal="box",
                            title=f"Distribution of {hist_column}",
                            template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

@st.cache_data
def get_box_plot(df, box_column):
    fig_box = px.box(df, x=box_column, title="Box Plot by Category")
    st.plotly_chart(fig_box, use_container_width=True)

if app_mode == "Overview":
    st.title(" Smartphone Usage & Productivity Analysis")

    st.markdown("""
    ### Project Overview
    This project analyzes how smartphone usage patterns impact productivity levels.

    ### Objectives
    - Explore relationships between screen time, sleep, and productivity
    - Identify patterns that differentiate productivity groups
    - Evaluate model performance using accuracy and other metrics

    ### Dataset
    - 50,000 observations
    - Features include phone usage, sleep hours, age, and social media usage
    """)    


if app_mode == "EDA":
    st.header("Exploratory Data Analysis")
    st.markdown("Interactively explore relationships between variables.")
    
    # Preview
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write("Dataset Dimensions:", df.shape)

    # Summary
    st.subheader("Summary Statistics")
    st.write(df.describe())

    
    st.subheader("Scatter Plot")

   
    # --------------------------
    # Scatter Plot
    # --------------------------
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("Select X-axis", numeric_columns)
    y_axis = st.selectbox("Select Y-axis", numeric_columns)
    color = st.selectbox("Select color grouping", categorical_columns)

    fig_scatter = px.scatter(
        df, x=x_axis, y=y_axis, color=color,
        title=f"{y_axis} vs {x_axis}",
        template="plotly_dark"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # --------------------------
    # Histogram + Box Plot
    # --------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Histogram")
        hist_column = st.selectbox("Histogram Column", numeric_columns)
        color_col = st.selectbox("Color Group", categorical_columns)
        get_hist(df, color_col, hist_column)

    with col2:
        st.subheader("Box Plot")
        box_column = st.selectbox("Box Plot Column", numeric_columns)
        get_box_plot(df, box_column)

    st.markdown("---")

    # --------------------------
    # Correlation Matrix (sampled for speed)
    # --------------------------
    st.subheader("Correlation Matrix")

    df_sample = df.sample(5000, random_state=42)

    corr_fig = px.imshow(
        df_sample[numeric_columns].corr(),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    st.plotly_chart(corr_fig, use_container_width=True)


if app_mode == "Modeling & Evaluation":
    st.header("Modeling & Evaluation")

    st.subheader("Model Training")

    # Target variable
    target = st.selectbox("Select Target Variable", df.columns)

    # Features
    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include="object").columns.tolist()

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Model selection
    model_name = st.selectbox("Select Model", ["Decision Tree", "Random Forest"])
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=50, random_state=42)

    # Pipeline
    clf = Pipeline(steps=[("preprocessor", preprocessor),
                          ("model", model)])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # --------------------------
    # Metrics
    # --------------------------
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.3f}")
    st.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.3f}")

    st.markdown("---")

    # --------------------------
    # Confusion Matrix
    # --------------------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("---")

    # --------------------------
    # Feature Importance
    # --------------------------
    st.subheader("Feature Importance")
    if model_name == "Random Forest":
        try:
            feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
            importances = clf.named_steps["model"].feature_importances_
            feat_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)
            fig, ax = plt.subplots()
            sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax)
            ax.set_title("Top 10 Important Features")
            st.pyplot(fig)
        except:
            st.warning("Feature importance not available for this configuration.")



if app_mode == "Insights":
    st.header("Key Insights")

    st.markdown("""
    ## Findings
    - Increased phone usage is linked to lower productivity
    - Sleep has a strong positive impact on productivity
    - Social media usage contributes to decreased productivity

    ### Model Conclusion
    - Tree-based models captured non-linear relationships between features such as screen time, sleep, and social media usage.
    - Random Forest outperformed Decision Trees due to its ability to reduce variance and improve generalization.
   
     ### Limitations
    - The dataset may not capture external factors such as work environment or mental health.
    - Correlation does not imply causation; further analysis would be needed for causal inference.

    ### Final Takeaway
    Productivity is strongly influenced by daily behavioral patterns. Managing screen time and prioritizing sleep appear to be key drivers of improved outcomes.
""")

