import streamlit as st
import pandas as pd
from apputil import *

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')

st.title("Titanic Data Visualizations")

st.write(
'''
# Titanic Visualization 1
"Question: How does the survival rate vary across passenger class, sex, and age group on the Titanic?"
'''
)
summary_df = survival_demographics(df)
fig1 = visualize_demographic(summary_df)
st.plotly_chart(fig1, use_container_width=True)


st.write(
'''
# Titanic Visualization 2
"Question: How does family size relate to the average ticket fare across different passenger classes?"
'''
)
family_df = family_groups(df)
fig2 = visualize_families(family_df)
st.plotly_chart(fig2, use_container_width=True)


st.write(
'''
# Titanic Visualization Bonus
'''
)
# Generate and display the figure
fig3 = visualize_family_size()
st.plotly_chart(fig3, use_container_width=True)