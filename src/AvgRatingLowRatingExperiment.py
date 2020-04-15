from src import GetUserClimbData as gud
from src import UserClimbCollaborativeFiltering as cf

# Colorado data
print('Getting Data - Low')
path = '../user_ids/UserIDsBoulder.csv'
low_rating_df_with_id = gud.get_users_climbs(path)
low_rating_df = low_rating_df_with_id.drop(labels='user_id', axis=1)

print('Getting Data - Avg')
avg_rating_df_with_id = gud.get_users_climbs_average(path)
avg_rating_df = avg_rating_df_with_id.drop(labels='user_id', axis=1)

print('Evaluating low vs avg')
low_mae, low_mse = cf.eval_five_fold(5, low_rating_df)
avg_mae, avg_mse = cf.eval_five_fold(5, avg_rating_df)
