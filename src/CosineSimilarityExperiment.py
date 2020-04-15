import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src import GetUserClimbData as gud
from src import UserClimbCollaborativeFiltering as cf

# Colorado data
print('Getting Data')
path = '../user_ids/UserIDsBoulder.csv'
user_rating_df_with_id = gud.get_users_climbs(path)
user_rating_df = user_rating_df_with_id.drop(labels='user_id', axis=1)

# Compare pearson and cosine
print('Comparing Pearson and Cosine fifty times')
pearson_maes = []
pearson_mses = []
cosine_maes = []
cosine_mses = []
for i in range(0, 50):
    print('Comparison Number: ' + str(i))
    pearson_mae, pearson_mse = cf.eval_five_fold(5, user_rating_df)
    pearson_maes.append(pearson_mae)
    pearson_mses.append(pearson_mse)

    cosine_mae, cosine_mse = cf.eval_five_fold(5, user_rating_df, cosine_similarity)
    cosine_maes.append(cosine_mae)
    cosine_mses.append(cosine_mse)

avg_pearson_mae = np.mean(pearson_maes)
avg_pearson_mse = np.mean(pearson_mses)

avg_cosine_mae = np.mean(cosine_maes)
avg_cosine_mse = np.mean(cosine_mses)


