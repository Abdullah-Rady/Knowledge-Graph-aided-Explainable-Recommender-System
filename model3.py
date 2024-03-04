# Import necessary libraries
import pandas as pd
import numpy as np
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split

# Load the MovieLens 100k dataset
data = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Create a mapping of user and item IDs to indices
user_ids = data['user_id'].unique()
user2idx = {user_id: i for i, user_id in enumerate(user_ids)}
item_ids = data['item_id'].unique()
item2idx = {item_id: i for i, item_id in enumerate(item_ids)}

# Add columns to the DataFrame with the user and item indices
data['user_idx'] = data['user_id'].map(user2idx)
data['item_idx'] = data['item_id'].map(item2idx)

# Split the data into training and testing sets
train, test = train_test_split(data[['user_idx', 'item_idx', 'rating']], test_size=0.2, random_state=42)

# Define the embedding dimensions and input layers
embedding_dim = 10
num_users = len(user2idx)
num_items = len(item2idx)
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim)(user_input)
item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim)(item_input)

# Flatten the embeddings and compute the dot product
user_flattened = Flatten()(user_embedding)
item_flattened = Flatten()(item_embedding)
dot_product = Dot(axes=1)([user_flattened, item_flattened])

# Concatenate the dot product with additional features (if any)
concatenated = Concatenate()([dot_product])

# Add a fully connected layer with sigmoid activation for the final output
output = Dense(1, activation='sigmoid')(concatenated)

# Create the Keras model with the input and output layers
model = Model(inputs=[user_input, item_input], outputs=output)

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam')

# Train the model on the training set
model.fit(x=[train['user_idx'], train['item_idx']], y=train['rating'], epochs=10, batch_size=64, validation_data=([test['user_idx'], test['item_idx']], test['rating']))

# Define a function to get the top recommendations for a given user
def get_top_recommendations(user_id, num_recommendations=10):
    # Create a list of tuples (movie_id, predicted_rating) for all movies
    predictions = []
    for item_id in item_ids:
        item_idx = item2idx[item_id]
        predicted_rating = model.predict([np.array([user2idx[user_id]]), np.array([item_idx])])[0][0]
        predictions.append((item_id, predicted_rating))
    # Sort the list by predicted rating, in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)
    # Get the top recommended movies
    top_recommendations = [x[0] for x in predictions[:num_recommendations]]
    return top_recommendations

# Example usage: Get the top 10 movie recommendations for user 42
top_movies = get_top_recommendations(42, num_recommendations=10)
print("Top recommended movies for user 42:", top_movies)
