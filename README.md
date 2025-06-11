# amazon-alexa-review-analysis
# 🤖 Amazon Alexa Review Sentiment Classifier with Streamlit

This project is an interactive web app that predicts whether a customer review of Amazon Alexa is **positive** or **negative** using Machine Learning (NLP). Built using Python and Streamlit, it combines **Naive Bayes** and **Logistic Regression** in a voting ensemble model.

## 🎯 Objectives

- Build a sentiment classification model using product reviews.
- Clean and vectorize text data with NLP techniques.
- Train multiple ML models and ensemble them using `VotingClassifier`.
- Deploy the model with a simple, intuitive web interface via **Streamlit**.

## 🛠️ Tools & Technologies

- **Python**: pandas, numpy, nltk, scikit-learn
- **NLP**: stopwords removal, punctuation cleaning, `CountVectorizer`
- **ML Models**: Multinomial Naive Bayes, Logistic Regression, VotingClassifier
- **Deployment**: Streamlit

## 📁 File Structure

```
amazon-alexa-streamlit-app/
├── amazon_alexa.tsv                # Dataset (TSV format)
├── app.py                         # Main Streamlit app
├── README.md                      # Project description
├── requirements.txt               # Dependencies
```

## 🧠 How It Works

1. Clean review text (punctuation, stopwords)
2. Vectorize with `CountVectorizer`
3. Train Naive Bayes & Logistic Regression
4. Combine with `VotingClassifier`
5. Predict using Streamlit interface

## 🔍 Example Inputs

- `I love this product, it's amazing!` → ✅ Positive
- `This product is terrible and I hate it!` → ❌ Negative

## 📦 Data Source

[Amazon Alexa Reviews - Kaggle](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews)

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

🙋 About the Author
**Nguyễn Thị Trà My**
Email: myntt0825@gmail.com
🔗 GitHub: [github.com/Miit08](https://github.com/Miit08)
