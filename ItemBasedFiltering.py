#%%
from src import RequestMPData as mp
from src import GetUserClimbData as gud
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
# import numba
# import numba-scipy
# from pprint import pprint
# pprint(mp.get_routes([112775509,112775510,112775511]))

'''
TODO: 
(1) Get all routes initial group of uses have climbed
(2) use get route by lat long to get more routes
(3) use features returned from get_routes
'''

#%%
def get_similar_routes(route_num,route_data,sims):
    rvec = route_data.iloc[route_num]
    sim_routes = sims[route_num]
    pairs = [(route_data.iloc[other_route_num].name, sim)
        for other_route_num, sim in enumerate(sim_routes)
            if route_num != other_route_num and sim > 0]
    return sorted(pairs,
                key = lambda x: x[1],
                reverse = True)

#%%
def rec_climb_for_user(user_routes, user_id, route_users, sims, include_climbed_routes=False):
    '''
        Example
        recs_112233654 = rec_climb_for_user(uitems,112233654,rd,rd_cs)
        [(114149484, 17.336310683908188),
        (105760929, 16.496645076757396),
        (105749782, 16.415581300071086)]
    '''
    recs = defaultdict(float)
    user_climbs = user_routes.loc[user_id]
    # for each climb the user has done
    for i, rating in enumerate(user_climbs):
        # if they actually did it
        if rating >= 1.0:
            # get the routes similar to that one
            sim_routes = get_similar_routes(
                i,
                route_users,
                sims)
            # add up all the times we saw the routes
            for route, sim in sim_routes:
                recs[route] += sim

    recs = sorted(recs.items(),
        key = lambda x: x[1],
        reverse = True
    )
    if include_climbed_routes:
        return recs
    else:
        return [(route, sim) for route, sim in recs 
            if route not in user_climbs[user_climbs >= 1.0].index ]
    
#%%
from scipy.stats import pearsonr
#pearsonr = numba.njit(pearsonr)
#@numba.jit(nopython=True)
def calc_pearson(item_users):
    n_items = len(item_users)
    out = np.array([[
       pearsonr(item_users[i],item_users[j])[0] for j in range(n_items)] for i in range(n_items)])
    return out

# ex = np.array([[1,2,3,4],[4,3,2,1],[1,2,1,4]],dtype=np.float_)
# cp = calc_pearson(ex)
# rd_ps = calc_pearson(rd.values)
# print(rd_ps.shape)