#%%
from src import GetUserClimbData as gud
from ItemBasedFiltering import rec_climb_for_user
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#%%
path = './user_ids/uids.csv'
star_data = gud.get_users_climbs(path)
# print((star_data >= 1.0).sum(axis = 1))
uitems = star_data.set_index('user_id')
route_data = star_data.T  
rd = route_data.drop(labels='user_id', axis=0)
rd_cs = cosine_similarity(rd)
# rd_cs.shape

#%%
def trim_routes(users,ratio=0.5):
    for u in users.index:
        users.loc[u] 

#%%$
train = uitems[:25]
test = uitems[25:]
# recs = rec_climb_for_user(train,test.index[-1],rd,rd_cs)


# %%
