import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src import GetUserClimbData as gud
from src import UserClimbCollaborativeFiltering as cf

# Colorado data
print('Getting Data')
path = '../user_ids/UserIDsBoulder.csv'
user_rating_df_with_id = gud.get_users_climbs(path)
user_rating_df = user_rating_df_with_id.drop(labels='user_id', axis=1)

cf.eval_five_fold_ranking(29, user_rating_df, user_rating_df_with_id, '../galago/judgments.txt', '../galago/baseline.txt', 20)

