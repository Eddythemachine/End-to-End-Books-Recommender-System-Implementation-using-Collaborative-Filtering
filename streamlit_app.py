import os
import sys
import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu  # Install: pip install streamlit-option-menu
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException

# ---------------------------
# Recommendation Class (Enhanced)
# ---------------------------
class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    @st.cache_resource
    def load_book_pivot(self):
        return pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))

    @st.cache_resource
    def load_final_rating(self):
        return pickle.load(open(self.recommendation_config.final_rating_serialized_objects, 'rb'))

    @st.cache_resource
    def load_model(self):
        return pickle.load(open(self.recommendation_config.trained_model_path, 'rb'))

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
            
            # Calculate similarity scores
            similarity_scores = [1 - dist for dist in distances[0][1:]]
            
            return recommended_books, poster_urls, similarity_scores
        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            with st.spinner("Training the model... This may take a few minutes."):
                pipeline = TrainingPipeline()
                pipeline.start_training_pipeline()
            return True
        except Exception as e:
            raise AppException(e, sys) from e

    def get_popular_books(self, n=10):
        """Get popular books based on ratings"""
        try:
            final_rating = self.load_final_rating()
            popular = final_rating.sort_values('avg_rating', ascending=False).head(n)
            return popular[['title', 'image_url', 'avg_rating', 'num_ratings']]
        except Exception as e:
            raise AppException(e, sys) from e

# ---------------------------
# Custom CSS Styling
# ---------------------------
def load_css():
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Titles */
    .main-title {
        text-align: center;
        color: #1E3A8A;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .book-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .book-title {
        font-weight: bold;
        color: #1F2937;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        height: 3em;
        overflow: hidden;
    }
    
    .similarity-badge {
        display: inline-block;
        background: #10B981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .primary-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    .secondary-button {
        background: #F3F4F6;
        color: #374151;
        border: 1px solid #D1D5DB;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #667eea;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #F8FAFC;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px;
        padding: 0 1rem;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(
    page_title="📚 BookWise - AI Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = None

# Header
st.markdown('<h1 class="main-title">📚 BookWise</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover your next favorite book with AI-powered recommendations</p>', unsafe_allow_html=True)

# Initialize Recommendation engine
@st.cache_resource
def get_recommender():
    return Recommendation()

try:
    obj = get_recommender()
except Exception as e:
    st.error(f"Failed to initialize recommendation engine: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=100)
    st.markdown("### 🔍 Navigation")
    
    # Navigation menu
    selected_tab = option_menu(
        menu_title=None,
        options=["Home", "Recommend", "Popular", "Train"],
        icons=["house", "search", "star", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#667eea", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#667eea"},
        }
    )
    
    st.markdown("---")
    
    if selected_tab == "Recommend":
        st.markdown("### 📖 Select a Book")
        
        # Load book names with caching
        @st.cache_data
        def load_book_names():
            return pickle.load(open(os.path.join("templates", "book_names.pkl"), "rb"))
        
        book_names = load_book_names()
        
        # Searchable selectbox
        selected_book = st.selectbox(
            "Choose a book you like:",
            book_names,
            index=book_names.index(st.session_state.selected_book) if st.session_state.selected_book in book_names else 0,
            help="Type to search through the book list"
        )
        
        num_recs = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=10,
            value=5,
            help="How many recommendations would you like to see?"
        )
        
        # Update session state
        st.session_state.selected_book = selected_book
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🎯 Get Recommendations", use_container_width=True, type="primary"):
                st.session_state.get_recs = True
        with col2:
            if st.button("🔄 Clear", use_container_width=True, type="secondary"):
                st.session_state.recommendations = None
                st.session_state.get_recs = False
    
    elif selected_tab == "Train":
        st.markdown("### ⚙️ Model Training")
        st.info("""
        This will retrain the recommendation model with the latest data.
        Training time depends on your dataset size.
        """)
        
        if st.button("🚀 Start Training", use_container_width=True, type="primary"):
            with st.spinner("Training in progress..."):
                progress_bar = st.progress(0)
                
                # Simulate progress updates
                for i in range(100):
                    progress_bar.progress(i + 1)
                    # Add actual training progress here
                
                try:
                    obj.train_engine()
                    st.success("✅ Training completed successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")

# Main content based on selected tab
if selected_tab == "Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to BookWise! ✨
        
        Discover books tailored to your taste using our AI-powered recommendation system.
        
        ### How it works:
        1. **Browse** popular books or search for ones you've enjoyed
        2. **Select** a book you like from our extensive collection
        3. **Get** personalized recommendations based on collaborative filtering
        4. **Explore** new titles with similar themes and styles
        
        ### Features:
        - 📊 **Smart Recommendations**: Based on user preferences and ratings
        - 🌟 **Popular Picks**: See what others are reading
        - 🎯 **Personalized**: Tailored to your reading history
        - ⚡ **Fast & Accurate**: Real-time recommendations
        
        Ready to find your next favorite book?
        """)
        
        if st.button("🎯 Start Exploring →", type="primary"):
            st.session_state.selected_tab = "Recommend"
            st.rerun()
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        try:
            # Example stats - replace with actual data
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("Total Books", "10,000+")
                st.metric("Users", "50,000+")
            with stats_col2:
                st.metric("Avg Rating", "4.2 ★")
                st.metric("Genres", "50+")
        except:
            pass

elif selected_tab == "Recommend":
    if hasattr(st.session_state, 'get_recs') and st.session_state.get_recs:
        try:
            with st.spinner("Finding the perfect recommendations for you..."):
                recommended_books, poster_urls, similarity_scores = obj.recommend_books(
                    st.session_state.selected_book, 
                    n_recs=num_recs
                )
            
            # Store in session state
            st.session_state.recommendations = {
                'books': recommended_books,
                'posters': poster_urls,
                'scores': similarity_scores
            }
            st.session_state.get_recs = False
            
        except Exception as e:
            st.error(f"❌ Could not generate recommendations: {e}")
    
    # Display recommendations if available
    if st.session_state.recommendations:
        st.markdown(f"""
        ### 📚 Recommended for fans of: **{st.session_state.selected_book}**
        *Based on collaborative filtering with similarity scores*
        """)
        
        # Display recommendations in grid
        n_cols = min(4, num_recs)
        cols = st.columns(n_cols)
        
        for i, (book, poster, score) in enumerate(zip(
            st.session_state.recommendations['books'],
            st.session_state.recommendations['posters'],
            st.session_state.recommendations['scores']
        )):
            col_idx = i % n_cols
            with cols[col_idx]:
                with st.container():
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    
                    if poster:
                        st.image(poster, use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/150x200?text=No+Cover", use_column_width=True)
                    
                    st.markdown(f'<div class="book-title">{book}</div>', unsafe_allow_html=True)
                    
                    # Similarity score
                    similarity_percent = int(score * 100)
                    st.markdown(f'<div class="similarity-badge">{similarity_percent}% match</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional metrics
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_similarity = int(np.mean(st.session_state.recommendations['scores']) * 100)
            st.metric("Average Similarity", f"{avg_similarity}%")
        with col2:
            st.metric("Recommendations", num_recs)
        with col3:
            st.metric("Confidence", "High" if avg_similarity > 70 else "Medium")

elif selected_tab == "Popular":
    st.markdown("### 🌟 Popular Books This Week")
    
    try:
        popular_books = obj.get_popular_books(8)
        
        # Display in grid
        cols = st.columns(4)
        for idx, (_, row) in enumerate(popular_books.iterrows()):
            with cols[idx % 4]:
                with st.container():
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    
                    if row['image_url']:
                        st.image(row['image_url'], use_column_width=True)
                    
                    st.markdown(f'<div class="book-title">{row["title"]}</div>', unsafe_allow_html=True)
                    
                    # Rating
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        st.markdown(f"⭐ **{row['avg_rating']:.1f}/5**")
                    with col2:
                        st.markdown(f"👥 {int(row['num_ratings'])}")
                    
                    if st.button(f"Get Similar", key=f"popular_{idx}", use_container_width=True):
                        st.session_state.selected_book = row['title']
                        st.session_state.selected_tab = "Recommend"
                        st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Could not load popular books: {e}")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
with footer_col1:
    st.markdown("""
    **BookWise** • AI-Powered Book Recommendations  
    [GitHub](https://github.com) • [Report Issue](https://github.com/issues)
    """)
with footer_col2:
    st.markdown("Powered by:")
    st.markdown("![Streamlit](https://streamlit.io/images/brand/streamlit-mark-color.svg)")
with footer_col3:
    st.markdown("""
    **Algorithms Used:**  
    • Collaborative Filtering  
    • K-Nearest Neighbors  
    • Matrix Factorization
    """)