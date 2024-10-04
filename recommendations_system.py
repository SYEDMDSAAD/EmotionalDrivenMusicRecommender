from surprise import SVD
import pandas as pd
import pickle

def load_model():
    with open('svd_model3.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model


def cf_recommendations(user_id, song_ids, model_cf):
    recommendations = []
    for song_id in song_ids:
        pred = model_cf.predict(user_id, song_id)
        recommendations.append((song_id, pred.est))  # song_id and predicted rating
    return sorted(recommendations, key=lambda x: x[1], reverse=True)




model = load_model()
