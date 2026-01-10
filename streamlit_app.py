import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException

# ---------------------------
# Recommendation Class
# ---------------------------
class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    @st.cache_resource
    def load_book_pivot(_self):
        return pickle.load(open(_self.recommendation_config.book_pivot_serialized_objects,'rb'))

    @st.cache_resource
    def load_final_rating(_self):
        return pickle.load(open(_self.recommendation_config.final_rating_serialized_objects,'rb'))

    @st.cache_resource
    def load_model(_self):
        return pickle.load(open(_self.recommendation_config.trained_model_path,'rb'))

    def fetch_posters(self, suggestion):
        try:
            book_pivot = self.load_book_pivot()
            final_rating = self.load_final_rating()

            poster_urls = []
            for book_id in suggestion[0]:
                book_name = book_pivot.index[book_id]
                idx = np.where(final_rating['title'] == book_name)[0][0]
                poster_urls.append(final_rating.iloc[idx]['image_url'])
            return poster_urls
        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_books(self, book_name, n_recs=5):
        try:
            model = self.load_model()
            book_pivot = self.load_book_pivot()

            book_id = np.where(book_pivot.index == book_name)[0][0]
            distances, suggestion = model.kneighbors(
                book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=n_recs + 1
            )

            recommended_books = [book_pivot.index[i] for i in suggestion[0]][1:]
            poster_urls = self.fetch_posters(suggestion)
            return recommended_books, poster_urls
        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            pipeline = TrainingPipeline()
            pipeline.start_training_pipeline()
            st.success("📈 Training Completed!")
            logging.info("Training pipeline executed successfully")
        except Exception as e:
            raise AppException(e, sys) from e

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(
    page_title="📚 Books Recommender",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
.stButton>button {
    color: white;
    background-color: #4CAF50;
    height: 3em;
    width: 100%;
    font-size: 16px;
    border-radius: 10px;
}
.stSelectbox>div>div>select {
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 End-to-End Books Recommender System")
st.write("This app uses collaborative filtering to recommend books based on your selection!")

# Initialize Recommendation engine
obj = Recommendation()

# Sidebar Controls
st.sidebar.header("Actions")
if st.sidebar.button("Train Recommender System"):
    obj.train_engine()

book_names = pickle.load(open(os.path.join("templates", "book_names.pkl"), "rb"))
selected_book = st.sidebar.selectbox("Select a book to get recommendations:", book_names)
num_recs = st.sidebar.slider("Number of recommendations to display:", min_value=1, max_value=10, value=5)

if st.sidebar.button("Get Recommendations"):
    try:
        recommended_books, poster_urls = obj.recommend_books(selected_book, n_recs=num_recs)

        st.subheader(f"Recommended books based on: **{selected_book}**")

        # Dynamic columns for recommendations
        n_cols = min(5, num_recs)
        cols = st.columns(n_cols)

        for i, (book, poster) in enumerate(zip(recommended_books, poster_urls)):
            col = cols[i % n_cols]
            col.markdown(f"**{book}**")
            if poster:
                col.image(poster, width=150)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.markdown("---")
st.write("Developed by: Your Name | Collaborative Filtering Demo | Streamlit App")
