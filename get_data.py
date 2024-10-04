import spotipy
from spotipy.oauth2 import SpotifyOAuth

CLIENT_ID = 'your-client-id'          # Replace with your Spotify Client ID
CLIENT_SECRET = 'your-client-secret'  # Replace with your Spotify Client Secret
REDIRECT_URI = 'your-redirect-uri'    # Replace with your Redirect URI

# Set the necessary scopes for accessing user's top tracks, saved tracks, and playlists
scope = 'user-top-read user-library-read user-read-recently-played playlist-read-private'

# Authenticate with Spotify using OAuth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=scope
))
def gather_implicit_data(sp):
    # Fetch user's recently played tracks (limit: 20)
    recent_tracks = sp.current_user_recently_played(limit=20)
    recent_track_ids = [track['track']['id'] for track in recent_tracks['items']]
    print("\nRecently Played Tracks:")
    for idx, track in enumerate(recent_tracks['items']):
        print(f"{idx+1}. {track['track']['name']} by {track['track']['artists'][0]['name']}")

    # Fetch user's top tracks (limit: 20)
    top_tracks = sp.current_user_top_tracks(limit=20, time_range='medium_term')
    top_track_ids = [track['id'] for track in top_tracks['items']]
    print("\nTop Tracks:")
    for idx, track in enumerate(top_tracks['items']):
        print(f"{idx+1}. {track['name']} by {track['artists'][0]['name']}")

    # Fetch user's saved (liked) tracks (limit: 20)
    saved_tracks = sp.current_user_saved_tracks(limit=20)
    saved_track_ids = [track['track']['id'] for track in saved_tracks['items']]
    print("\nSaved Tracks (Liked Songs):")
    for idx, item in enumerate(saved_tracks['items']):
        track = item['track']
        print(f"{idx+1}. {track['name']} by {track['artists'][0]['name']}")

    # Fetch user's playlists (limit: 10)
    playlists = sp.current_user_playlists(limit=10)
    print("\nUser Playlists:")
    for idx, playlist in enumerate(playlists['items']):
        print(f"{idx+1}. {playlist['name']} (total tracks: {playlist['tracks']['total']})")

    # Combine track IDs from recent, top, and saved tracks, and remove duplicates
    all_track_ids = list(set(recent_track_ids + top_track_ids + saved_track_ids))
    
    return all_track_ids

# Function to fetch audio features for a list of track IDs
def get_audio_features(sp, track_ids):
    audio_features = sp.audio_features(tracks=track_ids)

    print("\nAudio Features:")
    for idx, features in enumerate(audio_features):
        if features:  # Ensure the features exist (some tracks may not have audio features)
            print(f"Track {idx+1}: Danceability = {features['danceability']}, Energy = {features['energy']}, Valence = {features['valence']}")
        else:
            print(f"Track {idx+1}: No audio features available.")


# track_ids = gather_implicit_data(sp)
# print(f"\nAggregated Track IDs: {track_ids}")
# if track_ids:
#     get_audio_features(sp, track_ids)
# else:
#     print("No tracks found to extract audio features.")


