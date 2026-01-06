# Content-Based Movie Recommendation System

A machine learning project that suggests movies to users based on content similarity. This system analyzes movie metadata (genres, cast, director, keywords) to find and recommend movies similar to a user's favorite.

## 📌 Overview

A recommendation system is a technique used to suggest relevant items to users based on preferences. This project implements a **Content-Based Recommendation System**. 

Unlike systems that rely on what *other* users like (Collaborative Filtering), this system recommends items based on the **characteristics of the movies themselves**. If a user enjoys a specific movie, the system analyzes its features (like the director, genre, or keywords) and finds other movies that share those features.

## 🚀 Features

* **Data Processing:** Cleans and processes raw movie datasets.
* **Feature Engineering:** Combines critical metadata (Genres, Keywords, Tagline, Cast, Director) into a single "content profile" for each movie.
* **Vectorization:** Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text data into numerical vectors.
* **Similarity Search:** Calculates **Cosine Similarity** to mathematically measure the closeness between movies.
* **Fuzzy Matching:** Handles spelling errors in user input using `difflib` to find the closest matching movie title.

## 📂 Project Structure

```bash
├── data/
│   └── raw/
│       └── movies.csv      # The dataset used for recommendations
├── notebooks/
│   └── movie_recommender.ipynb  # The main Jupyter Notebook containing the code
├── requirements.txt        # List of python dependencies
└── README.md               # Project documentation

```

## 🛠️ Installation & Setup

Follow these steps to set up the project on your local machine.

### 1. Prerequisite: Python

Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

### 2. Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```

### 3. Create a Virtual Environment (Recommended)

It is best practice to use a virtual environment to manage dependencies.

* **Windows:**
```bash
python -m venv venv
venv\Scripts\activate

```


* **Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate

```



### 4. Install Dependencies

This project uses `requirements.txt` to install necessary libraries (pandas, numpy, scikit-learn, etc.). Run:

```bash
pip install -r requirements.txt

```

### 5. Launch Jupyter Notebook

To view and run the code, start the Jupyter server:

```bash
jupyter notebook

```

This will open your web browser. Navigate to the `notebooks` folder and open `movie_recommender.ipynb`.

## 🧠 How It Works (Step-by-Step)

### 1. Data Collection

We load the `movies.csv` file using `pandas`. This contains metadata like budget, genres, homepage, keywords, etc.

### 2. Feature Selection

We strictly select features that capture the **semantic** meaning of a movie. We ignore numerical data (like budget) in favor of text data that describes *what* the movie is:

* **Genres** (e.g., Action, Thriller)
* **Keywords** (Specific plot elements)
* **Tagline** (Short descriptions)
* **Cast & Director** (Key people involved)

### 3. Data Preprocessing

* **Filling Missing Values:** We replace `NaN` (empty) values with empty strings to prevent errors during text processing.
* **Combination:** We merge all selected features into a single string column called `combined_features`. This creates a "text profile" for every movie.

### 4. Vectorization (TF-IDF)

Computers cannot understand text directly. We use **TF-IDF Vectorizer** to convert the `combined_features` text into numbers.

* **TF (Term Frequency):** How often a word appears in a specific movie.
* **IDF (Inverse Document Frequency):** Reduces the weight of common words (like "the", "a") and increases the weight of unique words (like "Avatar" or "Nolan").

### 5. Similarity Calculation

We calculate the **Cosine Similarity** between all movie vectors.

* This creates a matrix where every movie is compared to every other movie.
* A score of 1.0 means identical; 0.0 means no similarity.

### 6. Recommendation

1. **Input:** The user types a movie name.
2. **Matching:** `difflib` finds the exact title in our database (fixing typos).
3. **Retrieval:** The system looks up the similarity scores for that movie.
4. **Sorting:** It sorts the scores in descending order.
5. **Output:** Prints the top 30 most similar movies.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📜 License

[MIT](https://choosealicense.com/licenses/mit/)

```

---

## Part 2: Detailed Explanation of Your Code

Here is the step-by-step breakdown of exactly what you did, why you did it, and the logic behind `requirements.txt`.

### 1. Importing Libraries
**Code:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

```

**Why:**

* **`pandas`:** To load the CSV file into a table (DataFrame) so you can manipulate columns easily.
* **`numpy`:** Used by pandas for numerical operations.
* **`sklearn (scikit-learn)`:** The core machine learning library. You used it for two specific things:
* `TfidfVectorizer`: To turn text into numbers.
* `cosine_similarity`: To calculate the math behind how "close" two movies are.


* **`difflib`:** A standard Python library used to compare strings. You needed this because users might make typos (e.g., typing "Avata" instead of "Avatar").

### 2. Data Loading & Feature Selection

**Code:**

```python
df = pd.read_csv("../data/raw/movies.csv")
selected_features = ["genres", "keywords", "tagline", "cast", "director"]

```

**Why:**
You have a massive dataset, but not all data helps recommend a movie.

* **Dropped:** `budget`, `vote_count`, etc. Two movies having the same budget doesn't mean a user will like both.
* **Kept:** `genres`, `cast`, `director`. These are **Content Features**. If you like *Interstellar* (Director: Christopher Nolan), you likely want to see *Inception* (Director: Christopher Nolan).

### 3. Handling Null Values (NaN)

**Code:**

```python
for features in selected_features:
    df[features] = df[features].fillna("")

```

**Why:**
Real-world data is messy. Some movies might not have a `tagline` or `keywords`. If you try to combine a string with a "Null" value, Python will crash or create errors. This loop fills holes with an empty string `""` so the code runs smoothly.

### 4. Creating the "Combined Feature"

**Code:**

```python
df["combined_features"] = combined_movies.apply(lambda row: " ".join(row.values.astype(str)), axis=1)

```

**Why:**
The algorithm needs **one** single block of text to analyze per movie. You cannot easily compare "Genre Column" vs "Genre Column" and "Cast Column" vs "Cast Column" separately and efficiently. By merging them, you create a "soup" of metadata.

* *Before:* Genre: "Action", Director: "James Cameron"
* *After:* "Action James Cameron ..."

### 5. Vectorization (The Math Part)

**Code:**

```python
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(df["combined_features"])

```

**Why:**
Computers can't read. They only understand math.

* **TF-IDF** looks at your "Combined Feature" string.
* If the word "Action" appears often in a movie, it gets a score.
* If the word "The" appears in *every* movie, TF-IDF lowers its score (because it's not unique).
* **Result:** `feature_vectors` is a giant matrix of numbers representing the "personality" of every movie.

### 6. Cosine Similarity

**Code:**

```python
similarity = cosine_similarity(feature_vectors)

```

**Why:**
You now have vectors (coordinates). Cosine similarity measures the angle between these vectors.

* Small angle = Very Similar (The vectors point in the same direction).
* Large angle = Not Similar.
This creates a grid (Matrix) where you can look up Movie A and see its score against Movie B, Movie C, etc.

### 7. The User Interaction (Difflib)

**Code:**

```python
find_close_match = difflib.get_close_matches(movie_name, list_of_all_movies)

```

**Why:**
If the user inputs "iron man" (lowercase) or "Iron Mann" (typo), exact matching fails. `difflib` finds the closest string in your database to ensure the code doesn't crash on slightly wrong inputs.

### 8. Retrieving and Sorting

**Code:**

```python
sorted_similar_movies = sorted(similarity_score, key=lambda x:x[1], reverse=True)

```

**Why:**

* `similarity_score` gives you a list of tuples like `(Movie_Index, Score)`.
* You need to show the **best** matches first, so you sort by the Score (`x[1]`) in descending order (`reverse=True`).

---

## Part 3: What is `requirements.txt` and Why Use It?

You asked how to explain the `requirements.txt` installation.

### What is it?

A `requirements.txt` file is a standard way in Python to list all the external libraries (dependencies) your project needs to run.

### Why do you need it?

You have `sklearn`, `pandas`, and `numpy` installed on *your* PC. If your friend clones your code, they might not have them. If they try to run your code, it will crash with `ModuleNotFoundError`.

### How to create it?

Since you have everything working on your PC, you can generate this file automatically by running this in your terminal:

```bash
pip freeze > requirements.txt

```

### How does the user use it?

When the user runs:

```bash
pip install -r requirements.txt

```

`pip` reads the file and automatically downloads and installs every library listed there at the correct version. This ensures their computer is exactly like yours.