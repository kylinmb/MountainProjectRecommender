from src import GetUserClimbData as gud
from src import UserClimbCollaborativeFiltering as cf
from sklearn.metrics.pairwise import cosine_similarity

# Colorado data
print('Getting Data - Low')
path = '../user_ids/UserIDsBoulder.csv'
low_rating_df_with_id = gud.get_users_climbs(path)
low_rating_df = low_rating_df_with_id.drop(labels='user_id', axis=1)

print('Getting Data - Avg')
avg_rating_df_with_id = gud.get_users_climbs_average(path)
avg_rating_df = avg_rating_df_with_id.drop(labels='user_id', axis=1)

print('Evaluating low vs avg')
low_mae_cos, low_mse_cos = cf.eval_five_fold(5, low_rating_df, cosine_similarity)
low_mae_pear, low_mse_pear = cf.eval_five_fold(5, low_rating_df)
avg_mae_cos, avg_mse_cos = cf.eval_five_fold(5, avg_rating_df, cosine_similarity)
avg_mae_pear, avg_mse_pear = cf.eval_five_fold(5, avg_rating_df)

cf.eval_five_fold_ranking(29, low_rating_df, low_rating_df_with_id, '../galago/low_cos_judgments_return10_', '../galago/low_cos_baseline_return10_', 10, cosine_similarity)
cf.eval_five_fold_ranking(29, avg_rating_df, avg_rating_df_with_id, '../galago/avg_cos_judgments_return10_', '../galago/avg_cos_baseline_return10_', 10, cosine_similarity)
cf.eval_five_fold_ranking(29, low_rating_df, low_rating_df_with_id, '../galago/low_pear_judgments', '../galago/low_pear_baseline', 20)
cf.eval_five_fold_ranking(29, avg_rating_df, avg_rating_df_with_id, '../galago/avg_pear_judgments', '../galago/avg_pear_baseline', 20)
