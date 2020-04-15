from src import GetUserClimbData as gud
from src import UserClimbCollaborativeFiltering as cf

# Colorado data
print('Getting Data')
path = '../user_ids/UserIDsBoulder.csv'
rating_df_with_id = gud.get_users_climbs(path)
rating_df = rating_df_with_id.drop('user_id', axis=1)
similarity = cf.pairwise_pearson_correlation(rating_df.to_numpy())
knn = cf.find_k_similar_neighbors(29, similarity)