import plotly.express as px
import pandas as pd

# update/add code below ...

'''Exercise 1: Create a function to group passengers'''

import plotly.express as px
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')

import pandas as pd

def survival_demographics(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Define age bins and labels
    age_bins = [0, 12, 19, 59, float('inf')]
    age_labels = ["Child", "Teen", "Adult", "Senior"]

    # Create age category column (using category dtype for clarity)
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)
    df['age_group'] = df['age_group'].astype(pd.CategoricalDtype(categories=age_labels, ordered=True))

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

# Example curiosity question to place above your visualization function in app.py:
# st.write("Did children in third class have a higher survival rate than adults in second class?")

# Example usage:
table = survival_demographics('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
print(table)

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'notebook'  # or 'notebook_connected' or 'iframe' if 'notebook' does not work

def visualize_demographic(table):
    # Focus on survival rate for targeted age/class groups
    # Example: compare Children in 3rd class vs. Adults in 2nd class
    subset = table[((table['Pclass'] == 3) & (table['age_group'] == 'Child')) |
                   ((table['Pclass'] == 2) & (table['age_group'] == 'Adult'))]

    fig = px.bar(subset,
                 x='age_group',
                 y='survival_rate',
                 color='Pclass',
                 barmode='group',
                 hover_data=['n_passengers', 'n_survivors'],
                 labels={'survival_rate': 'Survival Rate', 'age_group': 'Age Group', 'Pclass': 'Passenger Class'},
                 title='Survival Rate: Children in 3rd Class vs Adults in 2nd Class'
    )

    # Creative options: try a grouped bar chart for all combinations
    # Or use px.parallel_categories for categorical trends
    # Uncomment for grouped bar by sex and age group:
    #
    # fig = px.bar(table, x='age_group', y='survival_rate', color='Sex', facet_col='Pclass',
    #              barmode='group', title='Survival Rate by Age, Sex, and Class')

    return fig



'''Exercise 2: Create a function to identify family groups'''

def family_groups(file_path):
    df = pd.read_csv(file_path)
    df['family_size'] = df['SibSp'] + df['Parch'] + 1

    grouped = df.groupby(['family_size', 'Pclass'], observed=True)
    results = grouped['PassengerId'].agg(n_passengers='count')
    # Do not call .to_frame() here, as results is already a DataFrame

    # Add other aggregated fare stats
    results['avg_fare'] = grouped['Fare'].mean()
    results['min_fare'] = grouped['Fare'].min()
    results['max_fare'] = grouped['Fare'].max()

    results = results.reset_index()
    results = results.sort_values(['Pclass', 'family_size'])
    return results

def last_names(file_path):
    df = pd.read_csv(file_path)
    # Extract last name before comma in 'Name'
    df['last_name'] = df['Name'].apply(lambda x: x.split(',')[0].strip())
    # Count occurrences of each last name
    last_name_counts = df['last_name'].value_counts()
    return last_name_counts

family_table = family_groups('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
print(family_table.head())

last_name_counts = last_names('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
print(last_name_counts.head())