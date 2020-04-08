import numpy as np
from src import GetUserClimbData as gud
from sklearn.metrics.pairwise import cosine_similarity

# Get data
path = '../user_ids/UserIDsBoulder.csv'
star_data = gud.get_users_climbs(path)

# Drop User ID
star_data_no_id = star_data.drop(labels='user_id', axis=1)
star_matrix = star_data_no_id.to_numpy()

# compute cosine similarity
cs = cosine_similarity(star_matrix)

# find k most similar users
