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

def survival_demographics(data: pd.DataFrame) -> pd.DataFrame:
    df = prepare_columns(data)
    df = classify_age_bracket(df)
    df = categorize_class(df)

    # Group by Pclass, Sex, and age_group
    grouped = df.groupby(['Pclass', 'Sex', 'age_group'], observed=True)

    # Aggregate counts and survival statistics
    results = grouped['Survived'].agg(
        n_passengers='count',
        n_survivors='sum',
        survival_rate='mean'
    ).reset_index()

    # Reorder for clarity: sort by class (ascending), sex (female first), age_group (ordered)
    sex_order = ['female', 'male']
    results['Sex'] = pd.Categorical(results['Sex'], categories=sex_order, ordered=True)
    results = results.sort_values(['Pclass', 'Sex', 'age_group'])
    return results

def visualize_demographic(data: pd.DataFrame):
    survival_data = survival_demographics(data)
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


# Exercise 2

def family_groups(data: pd.DataFrame) -> pd.DataFrame:
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

def last_names(data: pd.DataFrame) -> pd.Series:
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


def visualize_families(data: pd.DataFrame):
    family_data = family_groups(data)
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


def visualize_family_size(data: pd.DataFrame):
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
