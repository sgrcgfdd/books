import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import pandas as pd
import warnings
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')  # 忽略告警

# 加载模型
model = tf.keras.models.load_model('D:\\regression_model2_1.h5')
book_embedding_matrix = model.get_layer('Book-Embedding').get_weights()[0]
user_embedding_matrix = model.get_layer('User-Embedding').get_weights()[0]

# 计算余弦相似度
def recommend_books(user_id, top_n=5):
    user_vector = user_embedding_matrix[user_id]
    similarities = cosine_similarity([user_vector], book_embedding_matrix)
    similar_indices = similarities.argsort()[0][-top_n-1:-1][::-1]
    return similar_indices

# 计算图书之间的余弦相似度
book_similarity_matrix = cosine_similarity(book_embedding_matrix)

# 推荐相似图书
def recommend_similar_books(book_id, top_n=5):
    similar_indices = book_similarity_matrix[book_id].argsort()[-top_n-1:-1][::-1]
    return similar_indices

# 读取数据集
dataset = pd.read_csv('D:\\book_douban.csv')

def validate_input(user_id, book_id):
    if user_id < 0 or user_id >= user_embedding_matrix.shape[0]:
        raise ValueError(f"Invalid user_id: {user_id}")
    if book_id < 0 or book_id >= book_embedding_matrix.shape[0]:
        raise ValueError(f"Invalid book_id: {book_id}")


def recommend_books_by_rating(rating):
    predictions = model.predict([dataset.user_id, dataset.book_id])
    dataset['predicted_rating'] = predictions
    tolerance = 0.1
    filtered_books = dataset[(dataset['predicted_rating'] >= rating - tolerance) & (dataset['predicted_rating'] <= rating + tolerance)]
    if len(filtered_books) < 10:
        return filtered_books
    recommended_books = filtered_books.sample(n=10, random_state=42)
    return recommended_books

# 读取数据集
data = pd.read_csv('D:\\book_douban.csv')

# 确保数据集中包含书名、作者和出版日期列
required_columns = ['title', 'author', '出版时间']
for column in required_columns:
    if column not in data.columns:
        raise KeyError(f"数据集中缺少 '{column}' 列")

# 文本处理
data['title'] = data['title'].str.replace('[^\w\s]', '')
data['title'] = data['title'].apply(lambda x: ' '.join(jieba.lcut(x)))

# 使用TF-IDF向量化书名
vectorizer = TfidfVectorizer()
title_vectors = vectorizer.fit_transform(data['title'])

# 计算余弦相似度
cosine_sim = cosine_similarity(title_vectors)

# 推荐函数
def get_recommendations(keyword, cosine_sim=cosine_sim):
    keyword_processed = ' '.join(jieba.lcut(keyword))
    keyword_vector = vectorizer.transform([keyword_processed])
    sim_scores = cosine_similarity(keyword_vector, title_vectors).flatten()
    sim_scores_sorted = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    book_indices = [i[0] for i in sim_scores_sorted[1:11]]
    return data[['title', 'author', '出版时间']].iloc[book_indices]

# Streamlit应用
st.title('图书推荐系统')

# 基于用户嵌入的推荐
st.header('基于用户嵌入的推荐')
user_id = st.number_input('输入用户ID', min_value=0, value=0)
if st.button('为用户推荐图书'):
    recommended_books = recommend_books(user_id)
    st.write(f'为用户 {user_id} 推荐的图书ID: {recommended_books}')

# 基于图书嵌入的推荐
st.header('基于图书嵌入的推荐')
book_id = st.number_input('输入图书ID', min_value=0, value=0)
if st.button('为图书推荐相似图书'):
    similar_books = recommend_similar_books(book_id)
    st.write(f'与图书 {book_id} 相似的图书ID: {similar_books}')

# 基于评分的推荐
st.header('基于评分的推荐')
input_rating = st.number_input('输入评分值', min_value=0.0, max_value=5.0, value=3.0)
if st.button('为评分推荐图书'):
    recommended_books = recommend_books_by_rating(input_rating)
    st.write(f'推荐的书单（评分值为 {input_rating}):')
    st.write(recommended_books[['book_id', 'title', 'author']])

# 基于关键词的推荐
st.header('基于关键词的推荐')
keyword = st.text_input('输入关键词')
if st.button('为关键词推荐图书'):
    recommendations = get_recommendations(keyword)
    st.write(recommendations)