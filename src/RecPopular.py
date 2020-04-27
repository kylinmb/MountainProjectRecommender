from src.RequestMPData import get_routes
from src.AssesItemBased import trim_routes, format_for_galago, export_data, run_eval
from math import ceil, floor
import pandas as pd
# from multiprocessing import starmap,Pool

def batch_process_route_data(route_ids):
    divs = ceil(len(route_ids)/200)
    rf = pd.DataFrame()
    for i in range(divs):
        route_data = get_routes(route_ids[i*200:(i+1)*200])['routes']
        rf = rf.append(pd.DataFrame(route_data))
    return rf

def rate_popular_climbs(rf,starWieght=10,voteWeight=1):
    rf['rating'] = rf['stars'].astype(float)*starWieght \
        + rf['starVotes'].astype(int)*voteWeight
    return sorted(
        list(zip(rf['id'],rf['rating'])),
        key = lambda x: x[1],
        reverse = True)

def rec_popular_routes(user_data,rating_data,cutoff):
    recs = {}
    for u in list(user_data.index):
        not_climbed = list(user_data.loc[u][user_data.loc[u].astype(int) == 0].index.astype(int))
        recs[u] = [r for r in rating_data if r[0] in not_climbed][:cutoff]
    return recs

def test_folds_pop(user_data,rating_data,folds,cutoff,sw,vw):
    hold_out_size = floor(len(user_data)/folds)
    for i in range(hold_out_size - 1 ):
        test_split = user_data[ i*hold_out_size : (i+1)*hold_out_size ]
        train_split = user_data[0:i*hold_out_size].append(
            user_data[(i+1)*hold_out_size :])
        
        held_out_routes, test_trim = trim_routes(test_split,0.2)
        train_held_out = train_split.append(test_trim)

        recs = rec_popular_routes(user_data,rating_data,cutoff)
        # print('recs are', recs )

        judge, base = format_for_galago(recs,held_out_routes)
        export_data('popular_{}_{}_{}_'.format(sw,vw,cutoff),judge,base,i+1)

def run_pop(user_data,cutoff,sw,vw):
    rf = pd.read_csv('./data/route_data.csv')
    pop_climbs = rate_popular_climbs(rf,sw,vw)
    # print('pop climbs are ',pop_climbs)
    test_folds_pop(user_data,pop_climbs,5,cutoff,sw,vw)
    return run_eval('popular_{}_{}_{}_'.format(sw,vw,cutoff),5)

def run_many_pop(user_data,cutoff,sws,vws,n_process=6):
    data_array = []
    mf = pd.DataFrame()
    for sw in sws:
        for vw in vws:
            mf = mf.append(run_pop(user_data,cutoff,sw,vw))
    return mf

    