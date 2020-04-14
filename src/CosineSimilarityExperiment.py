import matplotlib.pyplot as plt
import numpy as np
from random import sample
from sklearn.metrics.pairwise import cosine_similarity
from src import GetUserClimbData as gud
from src import UserClimbCollaborativeFiltering as cf

# Colorado data
print('Getting Data')
path = '../user_ids/UserIDsBoulder.csv'
user_rating_df_with_id = gud.get_users_climbs(path)
user_rating_df = user_rating_df_with_id.drop(labels='user_id', axis=1)
user_rating_np = user_rating_df.to_numpy()

# Calculate similarity, determine nearest neighbors, make single prediction
print('Calculating Similarity and Nearest Neighbors')
pearson_similarity = cf.pairwise_pearson_correlation(user_rating_np)
pearson_nearest_neighbors = cf.find_k_similar_neighbors(5, pearson_similarity)
cosine_sim = cosine_similarity(user_rating_np)
cosine_nearest_neighbors = cf.find_k_similar_neighbors(cosine_sim)

