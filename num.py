import numpy as np
import pandas as pd
import ast
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# Load datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on 'title'
movies = movies.merge(credits, on='title', how='inner')

# Select only relevant columns
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies['overview'] = movies['overview'].fillna('')  # Fill missing overviews with empty string

# Convert JSON-like columns
def convert(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except (ValueError, TypeError):
        return ['Unknown']

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert3(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)[:3]]
    except (ValueError, TypeError):
        return ['Unknown']

movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(text):
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return i['name']
    except (ValueError, TypeError):
        return "Unknown"
    return "Unknown"

movies['crew'] = movies['crew'].apply(fetch_director)

# Convert overview to words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Convert lists to strings
movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x))
movies['keywords'] = movies['keywords'].apply(lambda x: ' '.join(x))
movies['cast'] = movies['cast'].apply(lambda x: ' '.join(x))
movies['crew'] = movies['crew'].apply(lambda x: ' '.join(x))

# Create 'tags' column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: ' '.join(x).lower())

# Select required columns
new_df = movies[['id', 'title', 'tags']].copy()

# Apply Lemmatization
lemmatizer = WordNetLemmatizer()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

# Vectorization & Cosine Similarity (Optimized)
cv = CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2), max_df=0.85, min_df=2)
vector = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vector)

# Debugging - Ensure correct shapes
print(f"Vector Shape: {vector.shape}")  # Should match (num_movies, num_features)
print(f"Similarity Matrix Shape: {similarity.shape}")  # Should be (num_movies, num_movies)

# Movie Recommendation Function (Handles Case Insensitivity)
def recommend(movie):
    movie_indices = new_df[new_df['title'].str.lower() == movie.lower()].index.tolist()
    
    if not movie_indices:
        return ["❌ Movie not found. Please check spelling or select a valid movie."]
    
    movie_index = movie_indices[0]  # Use first match in case of duplicates
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), key=lambda x: x[1], reverse=True)[1:11]

    return [new_df.iloc[i[0]].title for i in movie_list]

# Test Recommendation
print(recommend("Batman Begins"))

# Save Processed Data
pickle.dump(new_df, open('movies_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("✅ Successfully saved `movies_dict.pkl` and `similarity.pkl`!")
