# News Bias Detector

This tool allows you to compare media tone across sources using sentiment analysis (VADER + BERT) and a trained ML classifier.

## Features

- Fetches news from selected sources via NewsAPI
- Analyzes title and description sentiment using:
  - **VADER**: Rule-based sentiment tool
  - **BERT**: Transformer-based sentiment model
- Trains a simple ML model to predict stance based on headlines
- Interactive Streamlit dashboard with visualizations and downloadable results

## Setup Instructions

### 🪟 Windows

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
streamlit run bias_detector.py
```

### 🐧 Mac/Linux

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
streamlit run bias_detector.py
```

## Project Structure

- `bias_detector.py`: Main Streamlit app
- `requirements.txt`: Python dependencies
- `bias_results.csv`: Output file (generated after running)

## ML Components

This project implements:

- **VADER**: SentimentIntensityAnalyzer from `nltk`
- **BERT**: Sentiment pipeline using HuggingFace `transformers`
- **Logistic Regression Classifier**: Trained on article headlines using TF-IDF features to detect stance (Pro/Neutral/Against)

## Example Output

- Side-by-side sentiment comparisons
- Stance interpretation per source
- Classifier predictions + downloadable CSV
