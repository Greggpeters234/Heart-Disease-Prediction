from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

DATA_PATH = Path(__file__).resolve().parent / "datasets" / "heart_disease_cleaned.csv"

SEX_LABELS = {0: "Female", 1: "Male"}
CHEST_PAIN_LABELS = {
    1: "Typical Angina",
    2: "Atypical Angina",
    3: "Non-anginal Pain",
    4: "Asymptomatic",
}
BIN_LABELS = {0: "No", 1: "Yes"}

RISK_COLUMNS = [
    "hypertension",
    "exercise angina",
    "fasting blood sugar",
    "family history",
    "ST depression",
]


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "age",
        "resting systolic bp",
        "cholesterol",
        "ST depression",
        "target",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["sex", "chest pain type", "resting ECG", "ST slope", *RISK_COLUMNS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().copy()


@st.cache_resource
def train_model(df: pd.DataFrame):
    features = [
        "age",
        "sex",
        "chest pain type",
        "resting systolic bp",
        "cholesterol",
        "fasting blood sugar",
        "resting ECG",
        "exercise angina",
        "ST depression",
        "ST slope",
        "hypertension",
        "family history",
    ]
    X = df[features]
    y = df["target"].astype(int)

    numeric_features = ["age", "resting systolic bp", "cholesterol", "ST depression"]
    categorical_features = [c for c in features if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X, y)
    return model


def format_labels(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["sex_label"] = frame["sex"].round().astype(int).map(SEX_LABELS)
    frame["cp_label"] = frame["chest pain type"].round().astype(int).map(CHEST_PAIN_LABELS)
    frame["target_label"] = frame["target"].round().astype(int).map({0: "No Disease", 1: "Disease"})
    return frame


def build_overview_tab(df: pd.DataFrame) -> None:
    st.subheader("Clinical Overview Dashboard")

    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    selected_age = st.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))
    filtered = df[(df["age"] >= selected_age[0]) & (df["age"] <= selected_age[1])]

    c1, c2, c3, c4 = st.columns(4)
    disease_rate = (filtered["target"].mean() * 100) if not filtered.empty else 0
    with c1:
        st.metric("Patients", f"{len(filtered):,}")
    with c2:
        st.metric("Heart Disease Cases", f"{int(filtered['target'].sum()):,}")
    with c3:
        st.metric("Disease Rate", f"{disease_rate:.1f}%")
    with c4:
        st.metric("Avg Cholesterol", f"{filtered['cholesterol'].mean():.1f} mg/dL" if not filtered.empty else "0.0 mg/dL")

    labeled = format_labels(filtered)

    age_group = (
        labeled.assign(age_group=(labeled["age"] // 5 * 5).astype(int).astype(str) + "-" + ((labeled["age"] // 5 * 5) + 4).astype(int).astype(str))
        .groupby(["age_group", "target_label"], as_index=False)
        .agg(cases=("target", "count"), avg_cholesterol=("cholesterol", "mean"))
    )
    fig_age = px.bar(
        age_group,
        x="age_group",
        y="cases",
        color="target_label",
        barmode="stack",
        hover_data=["avg_cholesterol"],
        title="Heart Disease and Average Cholesterol by Age Distribution",
        labels={"age_group": "Age Group", "cases": "Cases", "target_label": "Condition"},
    )

    cp_sex = labeled.groupby(["cp_label", "sex_label", "target_label"], as_index=False).size()
    fig_cp = px.bar(
        cp_sex,
        x="cp_label",
        y="size",
        color="sex_label",
        facet_col="target_label",
        barmode="group",
        title="Heart Disease by Chest Pain Type and Sex",
        labels={"cp_label": "Chest Pain Type", "size": "Patients", "sex_label": "Sex"},
    )

    corr_cols = [
        "target",
        "age",
        "cholesterol",
        "resting systolic bp",
        "fasting blood sugar",
        "exercise angina",
        "ST depression",
        "hypertension",
        "family history",
    ]
    corr = filtered[corr_cols].corr(numeric_only=True)
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Matrix of Heart Disease and Common Features",
    )

    disease_only = filtered[filtered["target"] == 1]
    risk_values = {col: int((disease_only[col] > 0).sum()) for col in RISK_COLUMNS}
    risk_df = pd.DataFrame(
        {
            "Risk Factor": [
                "Hypertension",
                "Exercise Angina",
                "High Fasting Blood Sugar",
                "Family History",
                "ST Depression",
            ],
            "Patients": list(risk_values.values()),
        }
    )
    fig_risk = px.pie(
        risk_df,
        values="Patients",
        names="Risk Factor",
        hole=0.45,
        title="Common Risk Factors Among Patients With Heart Disease",
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(fig_age, use_container_width=True)
        st.plotly_chart(fig_corr, use_container_width=True)
    with col_right:
        st.plotly_chart(fig_cp, use_container_width=True)
        st.plotly_chart(fig_risk, use_container_width=True)


def build_prediction_tab(df: pd.DataFrame) -> None:
    st.subheader("Patient Risk Prediction")
    st.caption("Logistic Regression model trained on project cleaned dataset.")

    model = train_model(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: SEX_LABELS[x])
        cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], format_func=lambda x: CHEST_PAIN_LABELS[x])
        resting_bp = st.number_input("Resting Systolic BP", min_value=80, max_value=260, value=130)
    with col2:
        cholesterol = st.number_input("Cholesterol", min_value=80, max_value=700, value=240)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: BIN_LABELS[x])
        resting_ecg = st.selectbox("Resting ECG", options=[0, 1, 2])
        exercise_angina = st.selectbox("Exercise Angina", options=[0, 1], format_func=lambda x: BIN_LABELS[x])
    with col3:
        st_depression = st.number_input("ST Depression", min_value=0.0, max_value=8.0, value=1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", options=[1, 2, 3])
        hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: BIN_LABELS[x])
        family_history = st.selectbox("Family History", options=[0, 1], format_func=lambda x: BIN_LABELS[x])

    if st.button("Predict Risk"):
        input_row = pd.DataFrame(
            [
                {
                    "age": age,
                    "sex": sex,
                    "chest pain type": cp,
                    "resting systolic bp": resting_bp,
                    "cholesterol": cholesterol,
                    "fasting blood sugar": fasting_bs,
                    "resting ECG": resting_ecg,
                    "exercise angina": exercise_angina,
                    "ST depression": st_depression,
                    "ST slope": st_slope,
                    "hypertension": hypertension,
                    "family history": family_history,
                }
            ]
        )
        probability = float(model.predict_proba(input_row)[0][1])
        prediction = "High Risk" if probability >= 0.5 else "Low Risk"
        st.metric("Predicted Risk", prediction)
        st.progress(probability)
        st.write(f"Estimated probability of heart disease: **{probability * 100:.2f}%**")


def main() -> None:
    st.title("Heart Disease Prediction Dashboard")
    st.write("Interactive analytics and patient-level heart disease risk prediction.")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    df = load_data(DATA_PATH)
    if df.empty:
        st.error("The dataset is empty after cleaning. Please verify the source file.")
        st.stop()

    tab1, tab2 = st.tabs(["Dashboard", "Prediction"])
    with tab1:
        build_overview_tab(df)
    with tab2:
        build_prediction_tab(df)


if __name__ == "__main__":
    main()
