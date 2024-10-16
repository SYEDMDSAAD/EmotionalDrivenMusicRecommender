from surprise import SVD,Dataset,Reader
import pandas as pd
import spotipy
import get_data
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access the variables
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REDIRECT_URI = os.getenv('REDIRECT_URI')


# Set the necessary scopes for accessing user's top tracks, saved tracks, and playlists


# Authenticate with Spotify using OAuth
def authenticate():
    scope = 'user-top-read user-library-read user-read-recently-played playlist-read-private'
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=scope
    ))
    print("Authentication successful.")
    return sp
def get_tracks_for_emotion(emotion,df_songs):
    if emotion == "happy":
        valence_range = (0.6, 1.0)
        energy_range = (0.5, 1.0)
    elif emotion == "sad":
        valence_range = (0.0, 0.4)
        energy_range = (0.0, 0.5)
    elif emotion == "angry":
        valence_range = (0.0, 0.4)
        energy_range = (0.7, 1.0)
    elif emotion == "calm":
        valence_range = (0.5, 0.8)
        energy_range = (0.1, 0.4)
    else:
        valence_range = (0.0, 1.0)
        energy_range = (0.0, 1.0)
    filtered_songs = df_songs[
        (df_songs['valence'] >= valence_range[0]) & 
        (df_songs['valence'] <= valence_range[1]) & 
        (df_songs['energy'] >= energy_range[0]) & 
        (df_songs['energy'] <= energy_range[1])
    ]
    return filtered_songs

def create_model(data):
    
    threshold = data['play_count'].quantile(0.99)  # Remove top 1% outliers
    data = data[data['play_count'] < threshold]

    max_play_count = data['play_count'].max() 
    data['scaled_play_count'] = data['play_count'] / max_play_count * 5

    reader = Reader(rating_scale=(1, 5))
    reader = Reader()

    data = Dataset.load_from_df(data[['user_id', 'track_id', 'scaled_play_count']], reader)

    trainset = data.build_full_trainset()

    model_cf = SVD(n_factors=100, lr_all=0.01, reg_all=0.02)
    model_cf.fit(trainset)
    return model_cf


from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_matrix(df_songs):
    features = df_songs[['danceability', 'energy', 'valence', 'tempo', 
                         'acousticness', 'loudness', 'instrumentalness', 
                         'speechiness']]
    similarity_matrix = cosine_similarity(features)
    similarity_df = pd.DataFrame(similarity_matrix, index=df_songs['track_id'], columns=df_songs['track_id'])
    print("Similarity matrix created.")
    return similarity_df
def cf_recommendations(user_id, song_ids, model_cf):
    recommendations = []
    for song_id in song_ids:
        pred = model_cf.predict(user_id, song_id)
        recommendations.append((song_id, pred.est))
    return recommendations

def content_based_recommendations(user_id, df_interactions, similarity_df):
    # Get the songs that the user has interacted with
    user_songs = df_interactions[df_interactions['user_id'] == user_id]['track_id']
    
    # Initialize score array for each song in similarity_df
    song_scores = np.zeros(similarity_df.shape[0])
    
    # Accumulate scores for each song the user has interacted with
    for song_id in user_songs:
        if song_id in similarity_df.index:
            song_scores += similarity_df[song_id].values
    
    # Create a list of (track_id, score) based on the song_scores
    recommendations = []
    for i in range(len(song_scores)):
        if song_scores[i] > 0:  # Only consider songs with a positive score
            track_id = similarity_df.index[i]  # Assuming index corresponds to track_ids
            recommendations.append((track_id, song_scores[i]))

    # Sort recommendations by score in descending order
    return sorted(recommendations, key=lambda x: x[1], reverse=True)

def hybrid_recommendations(user_id, df_interactions, df_songs, model_cf, similarity_df, alpha=0.5):
    all_songs = df_songs['track_id'].values
    cf_recs = cf_recommendations(user_id, all_songs, model_cf)
    cf_scores = {song_id: score for song_id, score in cf_recs}  

    # Get content-based recommendations
    content_recs = content_based_recommendations(user_id, df_interactions, similarity_df)
    content_scores = {i: score for i, score in content_recs}

    hybrid_scores = {}
    for song_id in all_songs:
        cf_score = cf_scores.get(song_id, 0)
        content_score = content_scores.get(song_id, 0)
        hybrid_score = alpha * cf_score + (1 - alpha) * content_score
        hybrid_scores[song_id] = hybrid_score
    print("Songs recommended using hybrid approach")

    return hybrid_scores
def fetch_new_releases(sp, limit=20):
    new_releases = sp.new_releases(limit=limit)
    song_ids = []
    
    
    # Iterate over each album in the new releases
    for album in new_releases['albums']['items']:
        # Album ID
        album_id = album['id']
        # Album name
        album_name = album['name']
        
        # Fetch tracks for this album
        tracks = sp.album_tracks(album_id)  # Fetch tracks for the album
        for track in tracks['items']:
            song_ids.append(track['id'])
    print("fetched new releases")
    return song_ids
def recommend(sp,emotion,alpha=0.5):
    data = get_data.gather_user_data(sp)
    new_songs = fetch_new_releases(sp)
    df_songs = get_data.get_audio_features(sp, data['track_id'].to_list())
    df_new_songs = get_data.get_audio_features(sp, new_songs)
    # Combine the dataframes
    df_songs = pd.concat([df_songs, df_new_songs], ignore_index=True)
    model_cf = create_model(data)
    filtered_songs = get_tracks_for_emotion(emotion,df_songs)
    if filtered_songs.empty:
        print(f"No songs found for the emotion '{emotion}'.")
        return []
    similarity_df = compute_similarity_matrix(filtered_songs)
    user_id = sp.current_user()['id']
    recommendations = hybrid_recommendations(user_id, data,filtered_songs, model_cf, similarity_df, alpha)
    print("Recommended songs based on emotion")
    return recommendations
sp = authenticate()
recommendations=recommend(sp,"happy",0.5)
for i in recommendations.keys():
    print(sp.track(i)['name'])