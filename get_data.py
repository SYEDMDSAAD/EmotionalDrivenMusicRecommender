
import pandas as pd
def gather_user_data(sp):
    user_id =sp.current_user()['id']
    # Initialize lists to collect track data
    track_data = []

    # Fetch user's recently played tracks (limit: 30)
    recent_tracks = sp.current_user_recently_played(limit=30)
    for track in recent_tracks['items']:
        track_data.append((track['track']['id'], 1))  # Each recent track gets a play count of 1

    # Fetch user's top tracks (limit: 30)
    top_tracks = sp.current_user_top_tracks(limit=30, time_range='medium_term')
    for track in top_tracks['items']:
        track_data.append((track['id'], 1))  # Each top track gets a play count of 1

    # Fetch user's saved (liked) tracks (limit: 30)
    saved_tracks = sp.current_user_saved_tracks(limit=30)
    for track in saved_tracks['items']:
        track_data.append((track['track']['id'], 1))  # Each saved track gets a play count of 1

    # Fetch user's playlists (limit: 10)
    playlists = sp.current_user_playlists(limit=10)
    for playlist in playlists['items']:
        tracks = sp.playlist_tracks(playlist['id'])
        for item in tracks['items']:
            if item['track']:
                track_data.append((item['track']['id'], 1))  # Each track in the playlist gets a play count of 1

    # Create a DataFrame from the collected data
    user_data = pd.DataFrame(track_data, columns=['track_id', 'play_count'])
    user_data['user_id'] = user_id  # Add the user_id column
    user_data = user_data.groupby(['user_id', 'track_id'], as_index=False).sum()  # Aggregate play counts

    return user_data
def get_audio_features(sp, track_ids):
    audio_features = []
    for i in range(50, len(track_ids), 50):
        chunk = track_ids[i:i + 50]  # Get the current chunk of 50 track IDs
        features = sp.audio_features(tracks=chunk)
        audio_features.extend(features)

    # Create a list to store the audio features
    features_list = []

    # Collect audio features into the list
    for features in audio_features:
        if features:  # Ensure the features exist
            features_dict = {
                'track_id': features['id'],
                'danceability': features['danceability'],
                'energy': features['energy'],
                'valence': features['valence'],
                'tempo': features['tempo'],
                'acousticness': features['acousticness'],
                'instrumentalness': features['instrumentalness'],
                'liveness': features['liveness'],
                'loudness': features['loudness'],
                'speechiness': features['speechiness'],
                'key': features['key'],
                'mode': features['mode'],
                'duration_ms': features['duration_ms'],
                'time_signature': features['time_signature']
            }
            features_list.append(features_dict)
        else:
            print("No audio features available for this track.")

    # Convert the list of features into a DataFrame
    audio_features_df = pd.DataFrame(features_list)

    print("\nAudio Features DataFrame:")
    print(audio_features_df)

    return audio_features_df




