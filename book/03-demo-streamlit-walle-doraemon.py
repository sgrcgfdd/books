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

dataset = pd.read_csv(r'book/data/book_douban2.csv')
def recommend_books_by_rating(rating):
    predictions = model.predict([dataset.user_id, dataset.book_id])
    dataset['predicted_rating'] = predictions
    tolerance = 0.1
    filtered_books = dataset[(dataset['predicted_rating'] >= rating - tolerance) & (dataset['predicted_rating'] <= rating + tolerance)]
    if len(filtered_books) < 10:
        return filtered_books
    recommended_books = filtered_books.sample(n=10, random_state=42)
    return recommended_books

# 推荐函数
def get_recommendations(keyword, cosine_sim=cosine_sim):
    keyword_processed = ' '.join(jieba.lcut(keyword))
    keyword_vector = vectorizer.transform([keyword_processed])
    sim_scores = cosine_similarity(keyword_vector, title_vectors).flatten()
    sim_scores_sorted = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    book_indices = [i[0] for i in sim_scores_sorted[1:6] if i[1] > 0]  # 只取相似度大于0的书籍
    if not book_indices:
        return "抱一丝咱书库里暂时没有呢要不换个词亲？"
    return data[['title', 'author', '出版社','出版时间']].iloc[book_indices]

# Streamlit应用
st.title('图书推荐系统')

# 基于关键词的推荐
st.header('基于大学生摸鱼无聊时灵光一现想到的第一个词的书籍推荐系统')
keyword = st.text_input('现在脑袋里在想什么词？在这里输入吧（ps：仅限一词，多的不会）')
if st.button('点我点我'):
    recommendations = get_recommendations(keyword)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write(recommendations)
        
# 基于评分的推荐
st.header('基于评分的推荐')
input_rating = st.number_input('输入评分值', min_value=0.0, max_value=5.0, value=3.0)
if st.button('为评分推荐图书'):
    recommended_books = recommend_books_by_rating(input_rating)
    st.write(f'推荐的书单（评分值为 {input_rating}):')
    st.write(recommended_books[['book_id', 'title', 'author']])

