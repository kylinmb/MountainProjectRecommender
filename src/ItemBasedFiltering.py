
from src import RequestMPData as mp
from src import GetUserClimbData as gud
import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict


def get_similar_routes(route_num,route_data,sims):
    rvec = route_data.iloc[route_num]
    sim_routes = sims[route_num]
    pairs = [(route_data.iloc[other_route_num].name, sim)
        for other_route_num, sim in enumerate(sim_routes)
            if route_num != other_route_num and sim > 0]
    return sorted(pairs,
                key = lambda x: x[1],
                reverse = True)


def rec_climb_for_user(user_routes, user_id, route_users, sims, cutoff=1000, include_climbed_routes=False):
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

    #recs = {route : val/10 for route, val in recs.items()}
    recs = sorted(recs.items(),
        key = lambda x: x[1],
        reverse = True
    )
    if include_climbed_routes:
        recs = recs[:cutoff]
    else:
        recs = [(route, sim) for route, sim in recs 
            if route not in user_climbs[user_climbs >= 1.0].index ][:cutoff]

    # rt = [val for key,val in recs.items()].sum()
    # recs_normed = {route : val/rt for route, val in recs.items()}

    return recs
    

