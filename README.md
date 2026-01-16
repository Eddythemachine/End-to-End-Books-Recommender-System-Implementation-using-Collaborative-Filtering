# End-to-End Book Recommendation System - Complete Guide

## 📚 Project Overview

This is a comprehensive **Book Recommendation System** built using **Collaborative Filtering** and deployed with **Streamlit**. The system analyzes user-book interactions to recommend similar books based on rating patterns.

## 🏗️ Project Structure

```
End-to-End-Books-Recommender-System-Implementation-using-Collaborative-Filtering/
│
├── artifacts/                    # Serialized model and data files
│   ├── model.pkl                # Trained NearestNeighbors model
│   ├── book_names.pkl           # Book titles for recommendations
│   ├── final_rating.pkl         # Processed rating data
│   └── book_pivot.pkl           # User-book rating matrix
│
├── config/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   └── config.yml.dockerignore  # Docker ignore config
│
├── data/raw/                    # Raw datasets
│   ├── BX-Books.csv            # Book metadata
│   ├── BX-Users.csv            # User information
│   └── BX-Book-Ratings.csv     # User-book ratings
│
├── notebook/                    # Jupyter notebooks for development
│   ├── research.ipynb          # Initial research and exploration
│   ├── practice.ipynb          # Practice and testing
│   └── start.ipynb             # Starting point notebook
│
├── templates/                   # HTML templates (if any)
│   └── book_names.pkl          # Book names for templates
│
├── main.py                     # Main application script
├── streamlit_app.py            # Streamlit deployment application
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup file
├── Dockerfile                  # Docker configuration
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore file
```

## 🛠️ Step-by-Step Implementation Guide

### 1. Setting Up Virtual Environment (Using Anaconda)

#### Option A: Using Conda
```bash
conda create -n book_recommender python=3.9 -y
conda activate book_recommender
cd End-to-End-Books-Recommender-System-Implementation-using-Collaborative-Filtering
```

#### Option B: Using Python venv
```bash
python -m venv venv
venv\Scripts\activate
source venv/bin/activate
```

## 2. Installing Dependencies

```bash
pip install -r requirements.txt
```

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn streamlit pickle-mixin
```

## 3. Data Processing Pipeline

### Data Loading and Cleaning
```python
books = pd.read_csv(
    "../data/raw/BX-Books.csv",
    sep=";",
    encoding="latin-1",
    on_bad_lines="skip",
    low_memory=False
)
```

### Data Filtering
```python
final_rating.drop_duplicates(["user_id", "title"], inplace=True)
```

## 4. Model Training

```python
from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm="brute")
model.fit(book_sparse)
```

## 5. Serialization

```python
import pickle
pickle.dump(model, open("../artifacts/model.pkl", "wb"))
```

## 🚀 Streamlit Deployment

```bash
streamlit run streamlit_app.py
```

## 📝 Summary

This project demonstrates a complete end-to-end machine learning pipeline using collaborative filtering and Streamlit deployment.
