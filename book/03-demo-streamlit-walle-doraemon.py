import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import pandas as pd
import warnings
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')  # 忽略告警

# 加载模型
model = tf.keras.models.load_model(r'book/model/regression_model2_1.h5')
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

# 读取新数据集
new_dataset = pd.read_csv(r'book/data/book_douban2.csv')

# 建立映射关系
book_info_map = new_dataset.set_index('book_id').to_dict(orient='index')

# 读取数据集
data = pd.read_csv(r'book/data/book_douban2.csv')

# 确保数据集中包含书名、作者和出版日期列
required_columns = ['title', 'author', '出版社','出版时间']
for column in required_columns:
    if column not in data.columns:
        raise KeyError(f"数据集中缺少 '{column}' 列")

# 文本处理
data['title'] = data['title'].str.replace('[^\w\s]', '')  # 去除标点符号
data['title'] = data['title'].apply(lambda x: ' '.join(jieba.lcut(x)))  # 分词

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
    book_indices = [i[0] for i in sim_scores_sorted[1:6] if i[1] > 0]  # 只取相似度大于0的书籍
    if not book_indices:
        return "当前数据库内没有相关书籍"
    return data[['title', 'author', '出版社','出版时间']].iloc[book_indices]

# Streamlit应用
st.title('图书推荐系统')

# 基于关键词的推荐
st.header('基于关键词的推荐')
keyword = st.text_input('输入关键词')
if st.button('为关键词推荐图书'):
    recommendations = get_recommendations(keyword)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write(recommendations)

