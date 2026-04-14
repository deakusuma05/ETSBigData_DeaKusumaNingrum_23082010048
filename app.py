import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title='DANA Sentiment Dashboard',
    layout='wide'
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv('df_combined_final.csv')

df = load_data()

# =========================
# PREPROCESS
# =========================
df['at'] = pd.to_datetime(df['at'], errors='coerce')

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎛️ Filters")

# DATE FILTER
min_date = df['at'].min()
max_date = df['at'].max()

date_range = st.sidebar.date_input(
    "📅 Select Date Range",
    [min_date, max_date]
)

# SENTIMENT FILTER
selected_sentiments = st.sidebar.multiselect(
    '😊 Sentiment',
    options=sorted(df['sentimen'].dropna().unique()),
    default=df['sentimen'].dropna().unique()
)

# CRITICAL FILTER
show_critical = st.sidebar.checkbox("⚠️ Show Only Critical (Rating ≤ 2)")

# APPLY FILTER
df_filtered = df[df['sentimen'].isin(selected_sentiments)]

if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_filtered[
        (df_filtered['at'] >= pd.to_datetime(start_date)) &
        (df_filtered['at'] <= pd.to_datetime(end_date))
    ]

if show_critical:
    df_filtered = df_filtered[df_filtered['score'] <= 2]

# =========================
# HEADER
# =========================
st.title("📊 DANA App Sentiment Monitoring Dashboard")

# =========================
# KPI
# =========================
st.subheader("📌 Key Performance Indicators")

total = len(df_filtered)
avg_rating = df_filtered['score'].mean() if total > 0 else 0

positive = (df_filtered['sentimen'] == 'positif').mean() * 100 if total > 0 else 0
critical = df_filtered[df_filtered['score'] <= 2].shape[0] / total * 100 if total > 0 else 0

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Reviews", f"{total:,}")
c2.metric("Avg Rating ⭐", f"{avg_rating:.2f}")
c3.metric("Positive Rate 😊", f"{positive:.2f}%")
c4.metric("Critical Rate 🚨", f"{critical:.2f}%")

# =========================
# VISUAL
# =========================
st.markdown("---")
st.subheader("📊 Sentiment Distribution")

sent_counts = df_filtered['sentimen'].value_counts()

fig1, ax1 = plt.subplots()
ax1.pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%')
st.pyplot(fig1)

# =========================
# STACKED BAR
# =========================
st.subheader("📊 Sentiment per Rating")

grouped = df_filtered.groupby(['score', 'sentimen']).size().unstack().fillna(0)
grouped.plot(kind='bar', stacked=True)

fig2 = plt.gcf()
st.pyplot(fig2)

# =========================
# CRITICAL ISSUE
# =========================
st.markdown("---")
st.subheader("🔥 Critical Issues")

saldo = df_filtered['content'].str.contains('saldo hilang', case=False, na=False).sum()
premium = df_filtered['content'].str.contains('premium', case=False, na=False).sum()

st.write(f"🚨 'Saldo hilang': {saldo}")
st.write(f"🛠️ 'Premium issue': {premium}")

# =========================
# PREDICTIVE
# =========================
st.markdown("---")
st.subheader("🤖 Predictive Analytics")

# SENTIMENT MODEL
X = df['content'].astype(str)
y = df['sentimen']

tfidf = TfidfVectorizer(max_features=2000)
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

acc1 = accuracy_score(y_test, model_nb.predict(X_test))
st.write(f"Naive Bayes Accuracy: {acc1:.2f}")

# CHURN MODEL
df['churn'] = df['score'].apply(lambda x: 1 if x <= 2 else 0)

y2 = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X_vec, y2, test_size=0.2)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

acc2 = accuracy_score(y_test, model_lr.predict(X_test))
st.write(f"Churn Model Accuracy: {acc2:.2f}")

# =========================
# DATA
# =========================
st.markdown("---")
st.subheader("📋 Data Preview")

st.dataframe(df_filtered[['userName', 'score', 'at', 'content', 'sentimen']].head(50))
