import streamlit as st
import pickle
import pandas as pd
import requests

# Load processed movies dictionary & convert to DataFrame
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)  # Convert dictionary back to DataFrame

# Load similarity matrix
similarity = pickle.load(open('similarity.pkl', 'rb'))


# Function to fetch movie poster from OMDb API
def fetch_poster(movie_title):
    api_key = "e3df209a"  # 🔹 Replace with your valid OMDb API key
    base_url = f"https://www.omdbapi.com/?apikey={api_key}&t={movie_title}"

    try:
        response = requests.get(base_url)
        data = response.json()

        if response.status_code == 200 and "Poster" in data:
            return data["Poster"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster: {e}")

    return None  # Return None if the poster is not found


# Function to recommend movies
def recommend(movie):
    try:
        movie_index = movies[movies['title'].str.lower() == movie.lower()].index[0]
        distance = similarity[movie_index]
        movie_list = sorted(list(enumerate(distance)), key=lambda x: x[1], reverse=True)[1:6]

        recommended_movies = []
        posters = []

        for i in movie_list:
            title = movies.iloc[i[0]].title
            recommended_movies.append(title)
            posters.append(fetch_poster(title))

        return recommended_movies, posters
    except IndexError:
        return [], []  # Return empty lists if movie is not found


# Streamlit UI
st.title('🎬 BrightMind Predictor')
st.write('Welcome to your AI-powered movie recommendation app!')

# Dropdown menu with movies
selected_movie_name = st.selectbox("Select a movie:", movies['title'].values)

# Search button
if st.button("Search"):
    recommendations, posters = recommend(selected_movie_name)

    if recommendations:
        st.subheader("Recommended Movies:")

        cols = st.columns(len(recommendations))  # Dynamically adjust columns
        for i, col in enumerate(cols):
            with col:
                st.text(recommendations[i])
                if posters[i]:  # Show poster if available
                    st.image(posters[i], use_column_width=True)
    else:
        st.warning("❌ No recommendations found. Please try another movie.")
