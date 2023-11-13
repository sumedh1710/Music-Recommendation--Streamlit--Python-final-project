import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load your dataset
spotify_data = pd.read_csv('C:\Dell\Desktop\python final project dataset\sampled_dataset.csv')  

# Normalize the attributes
scaler = MinMaxScaler()
spotify_data['danceability'] = scaler.fit_transform(spotify_data[['danceability']])
spotify_data['popularity'] = scaler.fit_transform(spotify_data[['popularity']])

# Select relevant features
features = ['danceability', 'popularity']

# Calculate item-item similarity
item_similarity = cosine_similarity(spotify_data[features])

# Streamlit UI
st.title("mv2.py")

# Input field for track ID
track_id = st.text_input("Enter Track ID:")

if st.button("Get Recommendations"):
    track_index = spotify_data[spotify_data['track_id'] == track_id].index
    if len(track_index) == 0:
        st.write(f"Track ID '{track_id}' not found in the dataset.")
    else:
        track_index = track_index[0]
        similar_indices = item_similarity[track_index].argsort()[::-1][1:11]
        recommendations = spotify_data.iloc[similar_indices][['track_id', 'track_name', 'artists']]
        st.subheader("Recommended Tracks:")
        st.write(recommendations)

        spotify_data.head()