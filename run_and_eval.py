import numpy as np
from src import GetUserClimbData as gud
from src import AssesItemBased as assess
from src import RecPopular as rp
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
import pandas as pd

print('Loading data...')
try:
    low_user_data = pd.read_csv('data/low_user_data.csv')
except:
    uid_path = './user_ids/uids.csv'
    low_user_data = gud.get_users_climbs(uid_path)
    low_user_data.to_csv('data/low_user_data.csv',index=False)

low_user_data_idx = low_user_data.set_index('user_id')
low_route_data = low_user_data.T  
low_route_data = low_route_data.drop(labels='user_id', axis=0)


try:
    avg_user_data = pd.read_csv('data/avg_user_data.csv')
except:
    uid_path = './user_ids/uids.csv'
    avg_user_data = gud.get_users_climbs_average(uid_path)
    avg_user_data.to_csv('data/avg_user_data.csv',index=False)

avg_user_data_idx = avg_user_data.set_index('user_id')
avg_route_data = avg_user_data.T  
avg_route_data = avg_route_data.drop(labels='user_id', axis=0)

nrf = pd.read_csv('data/normed_routes_with_feats.csv')

print('Calculating similarity...')
low_cos_sim = cosine_similarity(low_route_data)
avg_cos_sim = cosine_similarity(avg_route_data)
low_corr_sim = pairwise_distances(X=low_route_data.values,metric='correlation',n_jobs=6)
avg_corr_sim = pairwise_distances(X=avg_route_data.values,metric='correlation',n_jobs=6)

feat_cos_sim = cosine_similarity(nrf.values)

name = 'feat_cos_20_'
print('Running {}...'.format(name))
assess.test_folds(
    low_user_data_idx,
    feat_cos_sim,
    folds=5,
    cutoff=20,
    name=name
)
mf = assess.run_eval(
    name=name,
    folds=5
)
mf.to_csv('analyze/{}.csv'.format(name),index=False)

name = 'feat_cos_10_'
print('Running {}...'.format(name))
assess.test_folds(
    low_user_data_idx,
    feat_cos_sim,
    folds=5,
    cutoff=10,
    name=name
)
mf = assess.run_eval(
    name=name,
    folds=5
)
mf.to_csv('analyze/{}.csv'.format(name),index=False)

# print('Running popular (cutoff 10)...')
# mf = rp.run_many_pop(low_user_data_idx,10,[10,50,100],[0.5,1,2])
# mf.to_csv('analyze/low_pop_climbs_10.csv',index=False)
# mf = rp.run_many_pop(avg_user_data_idx,10,[10,50,100],[0.5,1,2])
# mf.to_csv('analyze/avg_pop_climbs_10.csv',index=False)

# print('Running popular (cutoff 20)...')
# mf = rp.run_many_pop(low_user_data_idx,20,[10,50,100],[0.5,1,2])
# mf.to_csv('analyze/low_pop_climbs_20.csv',index=False)
# mf = rp.run_many_pop(avg_user_data_idx,20,[10,50,100],[0.5,1,2])
# mf.to_csv('analyze/avg_pop_climbs_20.csv',index=False)

# name = 'low_corr_20_'
# print('Running {}...'.format(name))
# assess.test_folds(
#     low_user_data_idx,
#     low_corr_sim,
#     folds=5,
#     cutoff=20,
#     name=name
# )
# mf = assess.run_eval(
#     name=name,
#     folds=5
# )
# mf.to_csv('analyze/{}.csv'.format(name),index=False)

# name = 'low_corr_10_'
# print('Running {}...'.format(name))
# assess.test_folds(
#     low_user_data_idx,
#     low_corr_sim,
#     folds=5,
#     cutoff=10,
#     name=name
# )
# mf = assess.run_eval(
#     name=name,
#     folds=5
# )
# mf.to_csv('analyze/{}.csv'.format(name),index=False)

# name = 'avg_corr_20_'
# print('Running {}...'.format(name))
# assess.test_folds(
#     avg_user_data_idx,
#     avg_corr_sim,
#     folds=5,
#     cutoff=20,
#     name=name
# )
# mf = assess.run_eval(
#     name=name,
#     folds=5
# )
# mf.to_csv('analyze/{}.csv'.format(name),index=False)

# name = 'avg_corr_10_'
# print('Running {}...'.format(name))
# assess.test_folds(
#     avg_user_data_idx,
#     avg_corr_sim,
#     folds=5,
#     cutoff=10,
#     name=name
# )
# mf = assess.run_eval(
#     name=name,
#     folds=5
# )
# mf.to_csv('analyze/{}.csv'.format(name),index=False)

# name = 'low_cos_20_'
# print('Running {}...'.format(name))
# assess.test_folds(
#     low_user_data_idx,
#     low_cos_sim,
#     folds=5,
#     cutoff=20,
#     name=name
# )
# mf = assess.run_eval(
#     name=name,
#     folds=5
# )
# mf.to_csv('analyze/{}.csv'.format(name),index=False)

# name = 'low_cos_10_'
# print('Running {}...'.format(name))
# assess.test_folds(
#     low_user_data_idx,
#     low_cos_sim,
#     folds=5,
#     cutoff=10,
#     name=name
# )
# mf = assess.run_eval(
#     name=name,
#     folds=5
# )
# mf.to_csv('analyze/{}.csv'.format(name),index=False)

# name = 'avg_cos_20_'
# print('Running {}...'.format(name))
# assess.test_folds(
#     avg_user_data_idx,
#     avg_cos_sim,
#     folds=5,
#     cutoff=20,
#     name=name
# )
# mf = assess.run_eval(
#     name=name,
#     folds=5
# )
# mf.to_csv('analyze/{}.csv'.format(name),index=False)

# name = 'avg_cos_10_'
# print('Running {}...'.format(name))
# assess.test_folds(
#     avg_user_data_idx,
#     avg_cos_sim,
#     folds=5,
#     cutoff=10,
#     name=name
# )
# mf = assess.run_eval(
#     name=name,
#     folds=5
# )
# mf.to_csv('analyze/{}.csv'.format(name),index=False)