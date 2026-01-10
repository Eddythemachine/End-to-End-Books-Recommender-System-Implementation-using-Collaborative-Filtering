import os
import sys
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu

# Try to import your custom modules with error handling
try:
    from books_recommender.logger.log import logging
    from books_recommender.config.configuration import AppConfiguration
    from books_recommender.pipeline.training_pipeline import TrainingPipeline
    from books_recommender.exception.exception_handler import AppException
    CUSTOM_MODULES_AVAILABLE = True
except ImportError as e:
    st.warning(f"Custom modules not available: {e}")
    CUSTOM_MODULES_AVAILABLE = False
    # Define mock classes for testing
    class AppException(Exception):
        pass
    class AppConfiguration:
        def get_recommendation_config(self):
            class Config:
                book_pivot_serialized_objects = "artifacts/serialized_objects/book_pivot.pkl"
                final_rating_serialized_objects = "artifacts/serialized_objects/final_rating.pkl"
                trained_model_path = "artifacts/models/model.pkl"
            return Config()

# ---------------------------
# Recommendation Class
# ---------------------------
class Recommendation:
    def __init__(self, app_config=None):
        if app_config is None:
            app_config = AppConfiguration()
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            st.error(f"Configuration error: {e}")
            raise
    
    @st.cache_resource
    def load_book_pivot(self):
        try:
            return pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
        except Exception as e:
            st.error(f"Could not load book pivot: {e}")
            return None
    
    @st.cache_resource
    def load_final_rating(self):
        try:
            return pickle.load(open(self.recommendation_config.final_rating_serialized_objects, 'rb'))
        except Exception as e:
            st.error(f"Could not load final rating: {e}")
            return None
    
    @st.cache_resource
    def load_model(self):
        try:
            return pickle.load(open(self.recommendation_config.trained_model_path, 'rb'))
        except Exception as e:
            st.error(f"Could not load model: {e}")
            return None
    
    def recommend_books(self, book_name, n_recs=5):
        try:
            model = self.load_model()
            book_pivot = self.load_book_pivot()
            
            if book_name not in book_pivot.index:
                st.error(f"Book '{book_name}' not found in database")
                return [], [], []
            
            book_id = np.where(book_pivot.index == book_name)[0][0]
            distances, suggestion = model.kneighbors(
                book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=n_recs + 1
            )
            
            recommended_books = [book_pivot.index[i] for i in suggestion[0]][1:]
            similarity_scores = [1 - dist for dist in distances[0][1:]]
            
            # Get posters
            poster_urls = []
            final_rating = self.load_final_rating()
            for book in recommended_books:
                try:
                    idx = np.where(final_rating['title'] == book)[0][0]
                    poster_urls.append(final_rating.iloc[idx]['image_url'])
                except:
                    poster_urls.append(None)
            
            return recommended_books, poster_urls, similarity_scores
        except Exception as e:
            st.error(f"Recommendation error: {e}")
            return [], [], []

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(
    page_title="📚 Book Recommender",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book Recommendation System")
st.write("Discover books you'll love!")

# Initialize app
@st.cache_resource
def init_recommender():
    return Recommendation()

obj = init_recommender()

# Sidebar
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Home", "Recommend", "About"],
        icons=['house', 'search', 'info-circle'],
        menu_icon="cast",
        default_index=0
    )
    
    st.markdown("---")
    
    # Load sample book names
    try:
        if os.path.exists("templates/book_names.pkl"):
            book_names = pickle.load(open("templates/book_names.pkl", "rb"))
        else:
            # Create sample data for testing
            book_names = [
                "The Da Vinci Code", "Harry Potter and the Sorcerer's Stone",
                "The Hobbit", "1984", "To Kill a Mockingbird"
            ]
            st.info("Using sample book list")
    except:
        book_names = ["Book 1", "Book 2", "Book 3", "Book 4", "Book 5"]
        st.warning("Could not load book names, using sample")

if selected == "Home":
    st.markdown("""
    ## Welcome!
    
    This is a book recommendation system that suggests books based on collaborative filtering.
    
    ### How to use:
    1. Go to the **Recommend** tab
    2. Select a book you like
    3. Click "Get Recommendations"
    4. Discover new books!
    
    ### Features:
    - AI-powered recommendations
    - Similarity scores
    - Book cover images
    - Easy to use interface
    """)

elif selected == "Recommend":
    st.subheader("Get Book Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_book = st.selectbox("Select a book:", book_names)
    
    with col2:
        num_recs = st.slider("Number of recommendations:", 1, 10, 5)
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Finding recommendations..."):
            books, posters, scores = obj.recommend_books(selected_book, num_recs)
        
        if books:
            st.success(f"Found {len(books)} recommendations!")
            
            # Display in columns
            cols = st.columns(min(4, len(books)))
            for idx, (book, poster, score) in enumerate(zip(books, posters, scores)):
                with cols[idx % len(cols)]:
                    st.markdown(f"**{book}**")
                    st.markdown(f"*{int(score*100)}% match*")
                    if poster:
                        st.image(poster, width=150)
                    else:
                        st.image("https://via.placeholder.com/150x200?text=No+Cover", width=150)
        else:
            st.error("No recommendations found. Please try another book.")

elif selected == "About":
    st.markdown("""
    ## About This App
    
    **Book Recommendation System**
    
    Built with:
    - Streamlit for the web interface
    - Scikit-learn for machine learning
    - Collaborative filtering algorithm
    - K-Nearest Neighbors model
    
    ### How it works:
    1. The system analyzes user-book interactions
    2. Finds similar books based on user preferences
    3. Recommends books with high similarity scores
    
    ### Files needed:
    - `artifacts/models/model.pkl` - Trained model
    - `artifacts/serialized_objects/book_pivot.pkl` - Book data
    - `artifacts/serialized_objects/final_rating.pkl` - Ratings data
    - `templates/book_names.pkl` - Book names list
    """)

# Footer
st.markdown("---")
st.markdown("*Book Recommendation System*")