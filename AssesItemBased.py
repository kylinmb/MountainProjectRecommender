#%%
from src import GetUserClimbData as gud
from ItemBasedFiltering import rec_climb_for_user, get_similar_routes
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from math import floor
from collections import defaultdict

#%%
path = './user_ids/uids.csv'
star_data = gud.get_users_climbs(path)
# print((star_data >= 1.0).sum(axis = 1))
route_data = star_data.T  
rdidx= route_data.drop(labels='user_id', axis=0)
rd_cs = cosine_similarity(rd)
# rd_cs.shape

#%%
uitems = star_data.set_index('user_id')
train = uitems.copy()[:25]
test = uitems.copy()[25:]
def trim_routes(users,ratio=0.5):
    np.random.seed(30)
    out = defaultdict(float)
    for u in sorted(list(users.index)):
        climbed = users.copy().loc[u][users.loc[u] >= 1]
        # print(u,'climbed',len(climbed))
        random_climbs = np.random.choice(climbed.index, 
            size=floor(len(climbed)*ratio)
        )
        # print('selected',len(random_climbs),'random climbs')
        out[u] = list(zip(random_climbs,users.loc[u][random_climbs]))
        # print(out)
        # print(sum(users.loc[u][random_climbs] >= 1))
        init_user_has_climbed = sum(users.loc[u] >= 1)
        users.loc[u, random_climbs] = 0
        # print(len(users.loc[u, random_climbs]))
        # print(users)
    return out, users

real_routes, test = trim_routes(test)
utest = train.append(test)
utest.shape

#%%$
recs = rec_climb_for_user(
    utest,
    test.index[0],
    route_data,
    rd_cs)

# %%
len(sorted(real_routes[test.index[0]], key = lambda x: x[1], reverse=True))
recs[:len(real_routes[test.index[0]])]
# %%
