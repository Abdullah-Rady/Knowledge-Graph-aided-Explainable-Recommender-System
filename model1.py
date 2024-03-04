# Import necessary libraries
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import os

path = os.getcwd() + '/ml1m/'

# Load the MovieLens 100k dataset
data = pd.read_csv(f'{path}preproccessed/ratings.txt', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
print(data.head())

# Load the dataset into Surprise's reader format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

# Train the model using Singular Value Decomposition (SVD)
algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Fit the model on the entire dataset
trainset = data.build_full_trainset()
algo.fit(trainset)

# Get a list of all movie IDs
movie_ids = data.df['item_id'].unique()

# Define a function to get the top recommendations for a given user
def get_top_recommendations(user_id, num_recommendations=10):
    # Create a list of tuples (movie_id, predicted_rating) for all movies
    predictions = []
    for movie_id in movie_ids:
        predictions.append((movie_id, algo.predict(user_id, movie_id).est))
    # Sort the list by predicted rating, in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)
    # Get the top recommended movies
    top_recommendations = [x[0] for x in predictions[:num_recommendations]]
    return top_recommendations

# Example usage: Get the top 10 movie recommendations for user 42
top_movies = get_top_recommendations(42, num_recommendations=10)
print(top_movies)
