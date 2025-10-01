import plotly.express as px
import pandas as pd

# Function to analyze survival demographics
def survival_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze survival patterns on the Titanic by class, sex, and age group.

    Parameters:
        df (pd.DataFrame): Titanic dataset containing at least 'Age', 'Pclass', 'Sex', 'Survived'.

    Returns:
        pd.DataFrame: Summary table with Pclass, Sex, AgeGroup, n_passengers, n_survivors, survival_rate
    """

    # Step 1: Create age categories
    age_bins = [0, 12, 19, 59, float('inf')]
    age_labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)

    # Drop rows with missing AgeGroup (due to missing Age)
    df = df.dropna(subset=['AgeGroup'])

    # Step 2: Group by class, sex, and age group
    group_cols = ['Pclass', 'Sex', 'AgeGroup']
    grouped = df.groupby(group_cols)

    # Step 3: Aggregate total passengers and survivors
    result = grouped['Survived'].agg(
        n_passengers='count',
        n_survivors='sum'
    ).reset_index()

    # Step 4: Calculate survival rate
    result['survival_rate'] = result['n_survivors'] / result['n_passengers']

    # Step 5: Sort for readability
    result = result.sort_values(by=['Pclass', 'Sex', 'AgeGroup'])

    return result


# Function to visualize demographic summary
def visualize_demographic(summary_df: pd.DataFrame):
    """
    Create a Plotly bar chart showing survival rates across
    passenger class, sex, and age group.

    Parameters:
        summary_df (pd.DataFrame): Output of survival_demographics()

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly Figure object
    """
    fig = px.bar(
        summary_df,
        x="Pclass",
        y="survival_rate",
        color="AgeGroup",
        barmode="group",
        facet_col="Sex",
        category_orders={
            "Pclass": [1, 2, 3],
            "AgeGroup": ['Child', 'Teen', 'Adult', 'Senior'],
            "Sex": ["male", "female"]
        },
        labels={
            "Pclass": "Passenger Class",
            "survival_rate": "Survival Rate",
            "AgeGroup": "Age Group"
        },
        title="Survival Rate by Class, Sex, and Age Group"
    )

    fig.update_layout(
        yaxis=dict(tickformat=".0%"),
        legend_title_text='Age Group',
        height=500,
        template="plotly_white"
    )

    return fig


# Function to analyze family groups
def family_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a family_size column, groups by family size and class,
    and computes fare statistics.

    Parameters:
        df (pd.DataFrame): Titanic dataset

    Returns:
        pd.DataFrame: Grouped summary with passenger count and fare stats
    """

    # Ensure necessary columns are present
    required_cols = ['SibSp', 'Parch', 'Pclass', 'Fare']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")
        
    # Step 1: Add family_size = SibSp + Parch + 1 (self)
    df['family_size'] = df['SibSp'] + df['Parch'] + 1

    # Step 2: Group by family size and class
    grouped = df.groupby(['family_size', 'Pclass'])

    # Step 3: Aggregate values
    result = grouped['Fare'].agg(
        n_passengers='count',
        avg_fare='mean',
        min_fare='min',
        max_fare='max'
    ).reset_index()

    # Step 4: Sort for readability
    result = result.sort_values(by=['Pclass', 'family_size'])

    return result


# Function to extract and count last names
def last_names(df: pd.DataFrame) -> pd.Series:
    """
    Extracts last names from Name column and counts frequency.

    Parameters:
        df (pd.DataFrame): Titanic dataset

    Returns:
        pd.Series: Last name as index, count as values
    """

    # Extract last name before comma in Name
    df['LastName'] = df['Name'].apply(lambda name: name.split(',')[0].strip())

    # Count last names
    last_name_counts = df['LastName'].value_counts()

    return last_name_counts


# Visualization function for family groups
def visualize_families(family_df: pd.DataFrame):
    """
    Create a Plotly line chart showing the relationship between
    family size and average fare, broken down by passenger class.

    Parameters:
        family_df (pd.DataFrame): Output from family_groups()

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly Figure
    """
    fig = px.line(
        family_df,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        markers=True,
        labels={
            "family_size": "Family Size",
            "avg_fare": "Average Fare",
            "Pclass": "Passenger Class"
        },
        title="Average Fare by Family Size and Passenger Class"
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend_title_text='Class',
        height=500
    )
    return fig