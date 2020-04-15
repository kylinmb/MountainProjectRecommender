
from src import GetUserClimbData as gud
from src.ItemBasedFiltering import rec_climb_for_user, get_similar_routes
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from math import floor
from scipy.stats import rankdata
import os

def trim_routes(users,ratio=0.5):
    out = {}
    for u in sorted(list(users.index)):
        climbed = users.copy().loc[u][users.loc[u] >= 1]
        np.random.seed(30)
        random_climbs = np.random.choice(climbed.index, 
            size=floor(len(climbed)*ratio)
        )
        out[u] = sorted(
            list(zip(random_climbs,users.loc[u][random_climbs])),
            key = lambda x: x[1],
            reverse = True)
        users.loc[u, random_climbs] = 0
    return out, users

def get_predictions(user_data,test_users,route_data,sim,cutoff):
    recs = {}
    for u in sorted(list(test_users.index)):
        recs[u] = rec_climb_for_user(user_data,u,route_data,sim,cutoff=cutoff)
    return recs

def format_for_galago(user_recs,user_real):
    out_judge = []; out_base = []
    for user_id, real in user_real.items():
        for route, rating in real:
            out_judge.append(
                '{} 0 {} {:.0f}'.format(user_id,route,rating)
            )
    for user_id, recs in user_recs.items():
        rank_list = sorted(np.unique( [r[1] for r in recs ]),reverse=True)
        for route, score in recs:
            out_base.append(
                '{} 0 {} {} {:.4f}'.format(user_id,route,rank_list.index(score)+1,score)
            )

    return out_judge, out_base

def export_data(name,judge,base,fold):
    base_path = 'data/{}base_fold_{}.txt'.format(name,fold)
    judge_path = 'data/{}judge_fold_{}.txt'.format(name,fold)
    try:
        os.remove(base_path)
        os.remove(judge_path)
    except:
        pass
    np.savetxt(base_path,base, delimiter='\n',fmt='%s')
    np.savetxt(judge_path,judge, delimiter='\n',fmt='%s')

def test_folds(user_data,sim,folds,name,cutoff):
    hold_out_size = floor(len(user_data)/folds)
    for i in range(hold_out_size - 1 ):
        test_split = user_data[ i*hold_out_size : (i+1)*hold_out_size ]
        train_split = user_data[0:i*hold_out_size].append(
            user_data[(i+1)*hold_out_size :])
        
        held_out_routes, test_trim = trim_routes(test_split,0.2)
        train_held_out = train_split.append(test_trim)

        recs = get_predictions(
                train_held_out,
                test_split,
                train_held_out.T,
                sim,
                cutoff)

        judge, base = format_for_galago(recs,held_out_routes)
        export_data(name,judge,base,i+1)

def run_eval(name,folds):
    raw = []
    metrics = [[],[]]
    for i in range(folds):
        cmd = 'galago eval --judgments=data/{}judge_fold_{}.txt --baseline=data/{}base_fold_{}.txt --metrics+MAP --metrics+NDCG10 --metrics+num_rel --metrics+num_rel_ret'.format(name,i+1,name,i+1)
        stream = os.popen(cmd)
        out = stream.read()
        raw.append(
            [(l.split(' ')[0],l.split(' ')[-1]) for l in out.split('\n')[:-1]])
    names = np.array(raw).T[0]
    vals = np.array(raw).T[1]
    for i,metric in enumerate(names):
        flts = vals[i].astype(float)
        metrics[0].append('Mean {}'.format(metric[0]))
        metrics[0].append('Std {}'.format(metric[0]))
        metrics[1].append(round(np.mean(flts),5))
        metrics[1].append(round(np.std(flts),5))
    return metrics
    

##################
# old stuff
##################

def avg_precision(rec_ids,real_ids):
    out = 0
    rel_seen = 0
    for i,rec in enumerate(rec_ids):
        if rec in real_ids:
            rel_seen += 1
            out += rel_seen/(i+1)
    return out/len(real_ids)

def calc_map(recs,truth,test):
    ap = 0
    for u in sorted(list(test.index)):
        rec_ids = [r[0] for r in recs[u]]
        real_ids = [r[0] for r in truth[u]]
        ap += avg_precision(rec_ids,real_ids)
    return ap/len(test)

