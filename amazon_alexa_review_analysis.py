# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

# Tải stopwords
#nltk.download('stopwords')

# 1. Đọc dữ liệu
reviews_df = pd.read_csv('amazon_alexa.tsv', sep='\t')

# 2. Làm sạch dữ liệu
def message_cleaning(message):
    if isinstance(message, float):
        message = str(message)
    # Loại bỏ dấu câu
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    # Loại bỏ stopwords
    Test_punc_removed_join_clean = [
        word for word in Test_punc_removed_join.split() 
        if word.lower() not in stopwords.words('english')
    ]
    return ' '.join(Test_punc_removed_join_clean)

# Áp dụng làm sạch dữ liệu
reviews_df['cleaned_reviews'] = reviews_df['verified_reviews'].apply(message_cleaning)

# 3. Mã hóa từ
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews_df['cleaned_reviews']).toarray()
y = reviews_df['feedback']

# 4. Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Huấn luyện mô hình Naive Bayes
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

# 6. Huấn luyện mô hình Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# 7. Kết hợp mô hình Naive Bayes và Logistic Regression
voting_clf = VotingClassifier(estimators=[
    ('nb', NB_classifier),
    ('log_reg', log_reg)
], voting='hard')

voting_clf.fit(X_train, y_train)

# 8. Tạo giao diện Streamlit
st.title("Dự Đoán Sự Hài Lòng Của Khách Hàng")
st.write("Nhập một bình luận và chúng tôi sẽ dự đoán xem bình luận đó có tích cực hay không!")

# Nhập bình luận từ người dùng
user_input = st.text_area("Bình luận của bạn:")

if st.button("Dự đoán"):
    if user_input:
        # Làm sạch và mã hóa bình luận
        cleaned_input = message_cleaning(user_input)
        input_vector = vectorizer.transform([cleaned_input]).toarray()

        # Dự đoán
        prediction = voting_clf.predict(input_vector)

        # Hiển thị kết quả
        if prediction[0] == 1:
            st.success("Dự đoán: Tích cực!")
        else:
            st.error("Dự đoán: Tiêu cực!")
    else:
        st.warning("Vui lòng nhập một bình luận trước khi dự đoán.")
#I hate this terrible product
#I love this product, it's amazing!
#This product is terrible and I hate it!
