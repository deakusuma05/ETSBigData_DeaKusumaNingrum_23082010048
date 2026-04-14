import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os

# 1. Page Configuration
st.set_page_config(
    page_title='DANA App Sentiment Monitoring',
    page_icon='📊',
    layout='wide'
)

# Define Branding Colors
DANA_BLUE = '#1C85C7'

# 2. Data Loading
@st.cache_data
def load_data():
    # Load the processed combined dataset
    df = pd.read_csv('/content/df_combined_final.csv')
    return df

df = load_data()

# 3. Sidebar Branding & Filters
logo_path = '/content/logo dana.png'
if os.path.exists(logo_path):
    st.sidebar.image(Image.open(logo_path), width=200)

st.sidebar.title("Dashboard Filters")

# Sentiment Multi-select
sentiment_options = sorted(df['sentimen'].unique())
selected_sentiments = st.sidebar.multiselect(
    'Select Sentiment Label',
    options=sentiment_options,
    default=sentiment_options
)

# Rating Multi-select
rating_options = sorted(df['score'].unique())
selected_ratings = st.sidebar.multiselect(
    'Select Star Rating',
    options=rating_options,
    default=rating_options
)

# Apply Filters
df_filtered = df[
    (df['sentimen'].isin(selected_sentiments)) &
    (df['score'].isin(selected_ratings))
]

# 4. Header Section
st.markdown(f"""
    <div style='background-color:{DANA_BLUE};padding:15px;border-radius:10px;margin-bottom:25px;'>
        <h1 style='color:white;text-align:center;margin:0;'>DANA App Sentiment & Health Monitoring</h1>
    </div>
    """, unsafe_allow_html=True)

# 5. KPI Scorecards
st.subheader('Key Performance Indicators (KPIs)')
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric(label='Total Reviews', value='53,000')

with kpi_col2:
    st.metric(label='Avg Star Rating', value='3.74', delta='Target > 4.0', delta_color='inverse')

with kpi_col3:
    st.metric(label='Positive Sentiment Rate', value='54.58%', delta='Target > 80%', delta_color='inverse')

with kpi_col4:
    st.metric(label='Critical Rate (1-2 Stars)', value='28.02%', delta='Target < 10%', delta_color='inverse')

# 6. Visualizations Section
st.markdown('---')
st.subheader('Sentiment Analysis Visualizations')
vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    st.write('**Overall Sentiment Distribution (Donut Chart)**')
    # Distribution data derived from analysis
    sent_dist = pd.DataFrame({
        'Label': ['Positif', 'Negatif', 'Netral'],
        'Value': [54.58, 33.20, 12.22]
    })
    fig_donut = px.pie(
        sent_dist, values='Value', names='Label', hole=0.5,
        color='Label', color_discrete_map={'Positif': '#28A745', 'Negatif': '#DC3545', 'Netral': '#FFC107'}
    )
    fig_donut.update_layout(margin=dict(t=20, b=20, l=0, r=0))
    st.plotly_chart(fig_donut, use_container_width=True)

with vis_col2:
    st.write('**Sentiment Proportion per Rating (Stacked Bar)**')
    # Proportions data based on previous groupby results
    rating_prop = pd.DataFrame({
        'Rating': ['1', '2', '3', '4', '5'] * 3,
        'Sentimen': ['Negatif']*5 + ['Netral']*5 + ['Positif']*5,
        'Proportion': [0.74, 0.63, 0.42, 0.23, 0.15, 0.20, 0.25, 0.24, 0.11, 0.07, 0.06, 0.12, 0.34, 0.66, 0.78]
    })
    fig_stacked = px.bar(
        rating_prop, x='Rating', y='Proportion', color='Sentimen',
        color_discrete_map={'Positif': '#28A745', 'Negatif': '#DC3545', 'Netral': '#FFC107'},
        barmode='stack'
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

# 7. Critical Issue Tracking & Churn Risk
st.markdown('---')
st.subheader('Critical Issues & Churn Monitoring')
issue_col1, issue_col2 = st.columns(2)

with issue_col1:
    st.info('**Technical Keyword Frequency (KPI 6 & 7)**')
    st.write(f"- 🚨 **'saldo hilang'**: 346 occurrences")
    st.write(f"- 🛠️ **'premium' complaints**: 1,724 occurrences")
    st.caption("Target: 'saldo hilang' < 500, 'premium' < 300.")

with issue_col2:
    st.warning('**Predictive Churn Insights**')
    st.write("🎯 **High-Churn-Risk Users**: 296 identified")
    st.write("- *Persistent Critics*: 91 users")
    st.write("- *Declining Users (Drop >= 2 Stars)*: 227 users")
    st.button('Export Churn Risk Segment')

# 8. Data Preview
st.markdown('---')
st.subheader('Filtered Review Data')
st.dataframe(df_filtered[['userName', 'score', 'at', 'content', 'sentimen']].head(100), use_container_width=True)
