import streamlit as st
import pandas as pd
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Setup (run once if needed)
# nltk.download('vader_lexicon')
# nltk.download('punkt')

# Go to "https://newsapi.org/ and get an API"
api_key = 'YOUR API'
newsapi = NewsApiClient(api_key=api_key)
analyzer = SentimentIntensityAnalyzer()
bert_classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Utility Functions
@st.cache_data
def get_sources():
    response = newsapi.get_sources(language='en')
    return [(src['name'], src['id']) for src in response['sources']]

def fetch_articles(query, sources, page_size=5):
    all_articles = []
    for source in sources:
        try:
            res = newsapi.get_everything(q=query, sources=source, language='en',
                                         sort_by='relevancy', page_size=page_size)
            for article in res['articles']:
                all_articles.append({
                    'source': source,
                    'title': article['title'],
                    'description': article['description'],
                    'publishedAt': article['publishedAt'],
                    'url': article['url']
                })
        except Exception as e:
            st.warning(f"Error fetching from {source}: {e}")
    return pd.DataFrame(all_articles)

def get_vader_sentiment(text):
    if not text:
        return 0.0
    return analyzer.polarity_scores(text)['compound']

def interpret_score(score):
    if score >= 0.25:
        return "Pro/Supportive"
    elif score <= -0.25:
        return "Critical/Opposing"
    else:
        return "Neutral"

def get_bert_sentiment(text):
    if not text:
        return 'neutral'
    try:
        result = bert_classifier(text[:512])[0]
        return result['label'].lower()
    except:
        return 'neutral'

def merge_bert_sentiments(row):
    if row['bert_sentiment_title'] == row['bert_sentiment_desc']:
        return row['bert_sentiment_title']
    return 'mixed'

def train_classifier(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['title'])
    y = df['stance']
    if y.nunique() < 2:
        return None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    return clf, vectorizer, X_test, y_test

# Streamlit User Interface
st.title("News Bias Dashboard")
st.caption("Compare media tone using AI models (VADER + BERT) and machine learning stance prediction.")

st.markdown("""
**But Habibi, How Does it Work?**
- **VADER**: A rule-based sentiment tool tuned for social media and short text. Scores range from -1 (negative) to +1 (positive).
- **BERT**: A transformer-based model trained on tweets to detect 'positive', 'neutral', or 'negative' tones.
- Articles are fetched from selected news sources and analyzed on title + description. You’ll see both the **AI interpretation** and a **trained ML classifier’s** stance prediction.

If results seem too uniform, we’ll flag them as possibly **neutral or inconclusive topics**.
""")

sources_list = get_sources()
source_names = [name for name, _ in sources_list]
source_dict = dict(sources_list)

selected_sources = st.multiselect("Choose News Sources", source_names, default=source_names[:3])
query = st.text_input("Enter a search topic", value="Syria / Deir Ez-Zawr")
chart_type = st.radio("Select Chart Type", options=["VADER", "BERT"])

if st.button("Analyze"):
    if not selected_sources:
        st.error("Please select at least one news source.")
    else:
        with st.spinner("Fetching and analyzing articles..."):
            source_ids = [source_dict[name] for name in selected_sources]
            df = fetch_articles(query, source_ids)

            if df.empty:
                st.warning("No articles found. Try a different topic or source.")
            else:
                df['sentiment_title'] = df['title'].apply(get_vader_sentiment)
                df['sentiment_desc'] = df['description'].apply(get_vader_sentiment)
                df['sentiment_avg'] = df[['sentiment_title', 'sentiment_desc']].mean(axis=1)
                df['stance'] = df['sentiment_avg'].apply(interpret_score)

                df['bert_sentiment_title'] = df['title'].apply(get_bert_sentiment)
                df['bert_sentiment_desc'] = df['description'].apply(get_bert_sentiment)
                df['bert_sentiment_combined'] = df.apply(merge_bert_sentiments, axis=1)

                # Label mapping fix
                bert_label_map = {
                    "label_0": "Critical/Opposing",
                    "label_1": "Neutral",
                    "label_2": "Pro/Supportive"
                }
                df['bert_sentiment_combined'] = df['bert_sentiment_combined'].map(bert_label_map).fillna(df['bert_sentiment_combined'])

                if df['stance'].nunique() < 2:
                    st.warning("This topic appears to have low stance diversity. Content may be too neutral for clear bias detection.")

                clf, vectorizer, X_test, y_test = train_classifier(df)
                if clf:
                    X_all = vectorizer.transform(df['title'])
                    df['ML_Predicted_Stance'] = clf.predict(X_all)
                else:
                    df['ML_Predicted_Stance'] = 'N/A'

                # Chart
                st.subheader(f"🔍 {chart_type} Summary Chart")
                if chart_type == "VADER":
                    avg_sentiment = df.groupby('source')['sentiment_avg'].mean().reset_index()
                    avg_sentiment['stance'] = avg_sentiment['sentiment_avg'].apply(interpret_score)
                    fig, ax = plt.subplots(figsize=(9, 5))
                    sns.barplot(x='sentiment_avg', y='source', hue='stance', data=avg_sentiment,
                                dodge=False, palette='coolwarm', ax=ax)
                    ax.axvline(0, color='black', linestyle='--')
                    ax.set_xlabel('Average Sentiment Score (VADER)')
                    ax.set_title(f'Average Sentiment by Source for Topic: "{query}"')
                else:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    sns.countplot(y='source', hue='bert_sentiment_combined', data=df, palette='coolwarm', ax=ax)
                    ax.set_title(f'BERT Sentiment Labels by Source for Topic: "{query}"')
                    ax.set_xlabel("count")
                st.pyplot(fig)

                st.subheader("🗾 Detailed Output with ML Prediction")
                st.dataframe(df[['source', 'title', 'url', 'sentiment_avg', 'stance', 'bert_sentiment_combined', 'ML_Predicted_Stance']])

                st.download_button("Download CSV", df.to_csv(index=False), file_name="bias_results.csv")

                if clf:
                    st.subheader("Machine Learning Classifier Report")
                    y_pred = clf.predict(X_test)
                    st.text(classification_report(y_test, y_pred, zero_division=0))
