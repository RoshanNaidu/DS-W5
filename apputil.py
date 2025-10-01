import pandas as pd
import numpy as np
import re
import plotly.express as px


def prepare_columns(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    required_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Survived"]
    for c in required_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def classify_age_bracket(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    brackets = [-np.inf, 12, 19, 59, np.inf]
    labels = ["Child (<=12)", "Teen (13–19)", "Adult (20–59)", "Senior (60+)"]
    df["age_bracket"] = pd.cut(df["Age"], bins=brackets, labels=labels, include_lowest=True)
    df["age_bracket"] = df["age_bracket"].astype(pd.CategoricalDtype(categories=labels, ordered=True))
    return df


def categorize_class(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    if "Pclass" in df:
        df["Pclass"] = pd.Categorical(df["Pclass"], categories=[1, 2, 3], ordered=True)
    return df


def compute_survival_statistics(data: pd.DataFrame) -> pd.DataFrame:
    df = prepare_columns(data)
    df = classify_age_bracket(df)
    df = categorize_class(df)
    grouped = df.groupby(["Pclass", "Sex", "age_bracket"], dropna=False, observed=True)
    results = grouped["Survived"].agg(
        total_passengers="size",
        survivors=lambda x: np.nansum(x == 1)
    ).reset_index()
    results["survival_fraction"] = results["survivors"] / results["total_passengers"]
    results = results.sort_values(["Pclass", "Sex", "age_bracket"])
    return results.reset_index(drop=True)


def plot_survival_comparison(data: pd.DataFrame):
    survival_data = compute_survival_statistics(data)
    fig = px.bar(
        survival_data,
        x="age_bracket",
        y="survival_fraction",
        color="Sex",
        barmode="group",
        facet_col="Pclass",
        facet_col_wrap=3,
        category_orders={
            "age_bracket": ["Child (<=12)", "Teen (13–19)", "Adult (20–59)", "Senior (60+)"],
            "Pclass": [1, 2, 3]
        },
        labels={
            "age_bracket": "Age Category",
            "survival_fraction": "Survival Fraction",
            "Sex": "Gender",
            "Pclass": "Passenger Class"
        },
        title="Survival Rate by Age Category and Gender Across Classes"
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(legend_title_text="Gender", margin=dict(l=10, r=10, t=50, b=10), height=500, bargap=0.2)
    return fig


def compute_family_characteristics(data: pd.DataFrame) -> pd.DataFrame:
    df = prepare_columns(data)
    df = categorize_class(df)
    df["family_total"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    cleaned = df.dropna(subset=["family_total", "Pclass", "Fare"])
    grouped = (
        cleaned.groupby(["Pclass", "family_total"], observed=True)
        .agg(
            total_passengers=("Fare", "size"),
            average_fare=("Fare", "mean"),
            fare_minimum=("Fare", "min"),
            fare_maximum=("Fare", "max")
        )
        .reset_index()
        .sort_values(["Pclass", "family_total"])
    )
    return grouped


def extract_surnames(data: pd.DataFrame) -> pd.Series:
    if "Name" not in data:
        return pd.Series(dtype="int")
    df = data.copy()

    def parse_surname(name):
        if not isinstance(name, str):
            return np.nan
        surname = name.split(",")[0]
        surname = re.sub(r'["\'() ]+', " ", surname).strip()
        return surname

    surnames = df["Name"].map(parse_surname)
    return surnames.value_counts(dropna=True)


def plot_family_fare_relationship(data: pd.DataFrame):
    family_data = compute_family_characteristics(data)
    line_chart = px.line(
        family_data,
        x="family_total",
        y="average_fare",
        color="Pclass",
        markers=True,
        category_orders={"Pclass": [1, 2, 3]},
        labels={"family_total": "Family Size", "average_fare": "Average Ticket Fare", "Pclass": "Class"},
        title="Average Ticket Fare by Family Size and Passenger Class"
    )
    scatter_chart = px.scatter(
        family_data,
        x="family_total",
        y="average_fare",
        color="Pclass",
        size="total_passengers",
        category_orders={"Pclass": [1, 2, 3]},
        labels={"family_total": "Family Size", "average_fare": "Average Ticket Fare"}
    )
    for trace in scatter_chart.data:
        line_chart.add_trace(trace)
    line_chart.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=500)
    return line_chart


def add_age_classification(data: pd.DataFrame) -> pd.DataFrame:
    df = prepare_columns(data)
    df = categorize_class(df)
    median_ages = df.groupby("Pclass")["Age"].transform("median")
    df["older_than_class_median"] = (df["Age"] > median_ages) & df["Age"].notna()
    return df


def plot_age_class_survival(data: pd.DataFrame):
    df = add_age_classification(data)
    summary = (
        df.groupby(["Pclass", "Sex", "older_than_class_median"], observed=True)
        .agg(
            count=("Survived", "size"),
            survived_count=lambda x: np.nansum(x == 1)
        )
        .reset_index()
    )
    summary["survival_rate"] = summary["survived_count"] / summary["count"]
    sorted_summary = summary.sort_values(["Pclass", "Sex", "older_than_class_median"])
    fig = px.bar(
        sorted_summary,
        x="Sex",
        y="survival_rate",
        color="older_than_class_median",
        barmode="group",
        facet_col="Pclass",
        facet_col_wrap=3,
        category_orders={"Pclass": [1, 2, 3]},
        labels={
            "Sex": "Gender",
            "survival_rate": "Survival Rate",
            "older_than_class_median": "Older than Median Age in Class",
            "Pclass": "Passenger Class"
        },
        title="Survival by Gender and Age Group Within Classes"
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=500)
    return fig
# Note: The functions below are designed to be called from app.py
def visualize_demographic():
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    fig = plot_survival_comparison(df)
    return fig
