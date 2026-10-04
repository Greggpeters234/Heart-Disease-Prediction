import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "datasets/heart_disease_cleaned.csv"
FEATURE_COLUMNS = [
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

SEX_MAP = {0: "Female", 1: "Male"}
TARGET_MAP = {0: "No Heart Disease", 1: "Heart Disease"}
CHEST_PAIN_MAP = {
    1: "Typical Angina",
    2: "Atypical Angina",
    3: "Non-Anginal Pain",
    4: "Asymptomatic",
}


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_columns = [
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
        "target",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_columns).copy()
    df["sex_label"] = df["sex"].round().astype(int).map(SEX_MAP)
    df["target_label"] = df["target"].round().astype(int).map(TARGET_MAP)
    df["chest_pain_label"] = (
        df["chest pain type"].round().astype(int).map(CHEST_PAIN_MAP).fillna("Other")
    )
    return df


@st.cache_resource
def train_model(df: pd.DataFrame) -> Pipeline:
    x = df[FEATURE_COLUMNS]
    y = df["target"].round().astype(int)
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def add_styles() -> None:
    st.markdown(
        """
        <style>
            .main {background-color: #f7f9fc;}
            .block-container {padding-top: 1.2rem; max-width: 1200px;}
            .metric-card {
                background: white;
                padding: 1rem;
                border-radius: 0.8rem;
                border: 1px solid #e8eef7;
            }
            .subtitle {
                color: #2a4365;
                font-size: 0.95rem;
                margin-top: -0.5rem;
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    selected_age = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

    sex_options = sorted(df["sex_label"].dropna().unique().tolist())
    selected_sex = st.sidebar.multiselect("Sex", sex_options, default=sex_options)

    chest_options = sorted(df["chest_pain_label"].dropna().unique().tolist())
    selected_chest = st.sidebar.multiselect(
        "Chest pain type", chest_options, default=chest_options
    )

    target_options = sorted(df["target_label"].dropna().unique().tolist())
    selected_target = st.sidebar.multiselect(
        "Heart disease status", target_options, default=target_options
    )

    filtered = df[
        (df["age"].between(selected_age[0], selected_age[1]))
        & (df["sex_label"].isin(selected_sex))
        & (df["chest_pain_label"].isin(selected_chest))
        & (df["target_label"].isin(selected_target))
    ].copy()

    st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} rows")
    return filtered


def overview_tab(df: pd.DataFrame) -> None:
    c1, c2 = st.columns((1.2, 1), gap="large")

    with c1:
        fig_age = px.histogram(
            df,
            x="age",
            color="target_label",
            nbins=20,
            barmode="overlay",
            opacity=0.75,
            title="Age Distribution by Heart Disease Status",
            labels={"target_label": "Status", "age": "Age"},
            color_discrete_map={
                "Heart Disease": "#e53e3e",
                "No Heart Disease": "#3182ce",
            },
        )
        fig_age.update_layout(legend_title_text="")
        st.plotly_chart(fig_age, use_container_width=True)

    with c2:
        grouped = (
            df.groupby(["chest_pain_label", "target_label"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        fig_cp = px.bar(
            grouped,
            x="chest_pain_label",
            y="count",
            color="target_label",
            barmode="group",
            title="Chest Pain Types vs Heart Disease",
            labels={"chest_pain_label": "Chest Pain Type", "count": "Patients"},
            color_discrete_map={
                "Heart Disease": "#e53e3e",
                "No Heart Disease": "#3182ce",
            },
        )
        fig_cp.update_layout(xaxis_tickangle=-20, legend_title_text="")
        st.plotly_chart(fig_cp, use_container_width=True)


def risk_patterns_tab(df: pd.DataFrame) -> None:
    corr_cols = FEATURE_COLUMNS + ["target"]
    corr = df[corr_cols].corr(numeric_only=True)

    c1, c2 = st.columns((1.15, 1), gap="large")
    with c1:
        heat = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix",
        )
        heat.update_layout(margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(heat, use_container_width=True)

    with c2:
        positive = df[df["target"] == 1]
        if positive.empty:
            st.info("No heart disease-positive records available under current filters.")
            return

        risk_summary = pd.DataFrame(
            {
                "Risk Factor": ["Hypertension", "Family History", "Fasting Blood Sugar", "Exercise Angina"],
                "Percentage": [
                    positive["hypertension"].mean() * 100,
                    positive["family history"].mean() * 100,
                    positive["fasting blood sugar"].mean() * 100,
                    positive["exercise angina"].mean() * 100,
                ],
            }
        )
        risk_fig = px.bar(
            risk_summary,
            x="Risk Factor",
            y="Percentage",
            color="Percentage",
            color_continuous_scale="OrRd",
            title="Common Risk Factors in Positive Cases",
            labels={"Percentage": "Patients (%)"},
        )
        risk_fig.update_layout(showlegend=False)
        st.plotly_chart(risk_fig, use_container_width=True)


def prediction_tab(df: pd.DataFrame) -> None:
    model = train_model(df)
    st.write("Use the controls below to estimate heart disease risk probability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: SEX_MAP[x])
        chest_pain = st.selectbox(
            "Chest Pain Type", options=[1, 2, 3, 4], format_func=lambda x: CHEST_PAIN_MAP[x]
        )
        resting_bp = st.number_input("Resting Systolic BP", min_value=80, max_value=240, value=130)

    with col2:
        cholesterol = st.number_input("Cholesterol", min_value=80, max_value=700, value=240)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1])
        resting_ecg = st.selectbox("Resting ECG", options=[0, 1, 2])
        exercise_angina = st.selectbox("Exercise Angina", options=[0, 1])

    with col3:
        st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", options=[1, 2, 3])
        hypertension = st.selectbox("Hypertension", options=[0, 1])
        family_history = st.selectbox("Family History", options=[0, 1])

    input_df = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "chest pain type": chest_pain,
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

    if st.button("Estimate Risk", type="primary"):
        probability = float(model.predict_proba(input_df)[0][1])
        level = "High" if probability >= 0.5 else "Lower"
        c1, c2 = st.columns(2)
        c1.metric("Predicted Risk Probability", f"{probability * 100:.1f}%")
        c2.metric("Risk Category", level)

        gauge = px.bar_polar(
            r=[probability * 100, 100 - probability * 100],
            theta=["Risk", ""],
            color=["Risk", ""],
            color_discrete_map={"Risk": "#e53e3e", "": "#edf2f7"},
            title="Risk Indicator",
        )
        gauge.update_layout(showlegend=False)
        st.plotly_chart(gauge, use_container_width=True)


def data_tab(df: pd.DataFrame) -> None:
    st.download_button(
        "Download filtered data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_heart_disease_data.csv",
        mime="text/csv",
    )
    st.dataframe(df, use_container_width=True, height=420)


def main() -> None:
    st.set_page_config(
        page_title="Heart Disease Intelligence Dashboard",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    add_styles()

    st.title("Heart Disease Intelligence Dashboard")
    st.markdown(
        "<p class='subtitle'>Interactive analytics and risk exploration for heart disease prediction.</p>",
        unsafe_allow_html=True,
    )

    try:
        data = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()

    filtered = sidebar_filters(data)
    if filtered.empty:
        st.warning("No records match your filters. Adjust the filters in the sidebar.")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    prevalence = (filtered["target"].mean() * 100) if len(filtered) else 0
    m1.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    m1.metric("Patients", f"{len(filtered):,}")
    m1.markdown("</div>", unsafe_allow_html=True)

    m2.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    m2.metric("Heart Disease Rate", f"{prevalence:.1f}%")
    m2.markdown("</div>", unsafe_allow_html=True)

    m3.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    m3.metric("Avg. Cholesterol", f"{filtered['cholesterol'].mean():.1f}")
    m3.markdown("</div>", unsafe_allow_html=True)

    m4.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    m4.metric("Avg. Age", f"{filtered['age'].mean():.1f}")
    m4.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Risk Patterns", "Individual Check", "Data Explorer"]
    )

    with tab1:
        overview_tab(filtered)
    with tab2:
        risk_patterns_tab(filtered)
    with tab3:
        prediction_tab(data)
    with tab4:
        data_tab(filtered)


if __name__ == "__main__":
    main()
