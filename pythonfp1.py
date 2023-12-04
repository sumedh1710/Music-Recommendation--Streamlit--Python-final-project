import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


spotify_data = pd.read_csv('C:\Dell\Desktop\python final project dataset\sampled_dataset.csv')  


if len(spotify_data) == 0:
    st.error("The dataset is empty.")
else:
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(spotify_data[['danceability', 'popularity', 'energy', 'acousticness', 'instrumentalness']])
    spotify_data[['danceability', 'popularity', 'energy', 'acousticness', 'instrumentalness']] = scaled_features

    #Cosine similarity
    features = ['danceability', 'popularity', 'energy', 'acousticness', 'instrumentalness']

    #item-item similarity
    item_similarity = cosine_similarity(spotify_data[features])

    # Function to get recommendations
    def get_recommendations(track_id, item_similarity, num_recommendations=10):
        try:
            # Validate track_id
            track_id = str(track_id)  
            if track_id not in spotify_data['track_id'].values:
                raise ValueError(f"Track ID '{track_id}' not found in the dataset.")

            track_index = spotify_data[spotify_data['track_id'] == track_id].index
            track_index = track_index[0]
            similar_indices = item_similarity[track_index].argsort()[::-1][1:num_recommendations + 1]
            recommendations = spotify_data.iloc[similar_indices][['track_id', 'track_name', 'artists']]
            return recommendations

        except ValueError as e:
            st.error(str(e))
            return None

    # Streamlit app
    st.title("Spotify Track Recommender")

    # User input for track ID using dropdown with autocomplete
    selected_track_name = st.selectbox("Select a track:", spotify_data['track_name'].unique(), format_func=lambda x: x)
    track_id_input = spotify_data.loc[spotify_data['track_name'] == selected_track_name, 'track_id'].values[0]

    # Display recommendations only if a valid track ID is provided
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(track_id_input, item_similarity)
        if recommendations is not None:
            st.subheader("Top 10 Recommendations:")
            st.table(recommendations[['track_name', 'artists']])
