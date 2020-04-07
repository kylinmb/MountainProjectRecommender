import RequestMPData as mp
import pandas as pd


def get_users_climbs(path_to_user_ids):
    user_ids = pd.read_csv(path_to_user_ids, sep='\n', header=None)
    users_climbs = pd.DataFrame()
    for index, user_id in user_ids.iterrows():
        single_user_climb = pd.DataFrame()
        single_user_climb['user_id'] = user_id
        users_ticks = mp.get_ticks(user_id[0])['ticks']
        for t in users_ticks:
            route_id = t['routeId']
            user_stars = t['userStars']
            single_user_climb[route_id] = user_stars if not user_stars == -1 else 1
        users_climbs = pd.concat([users_climbs, single_user_climb], sort=False, ignore_index=True)
    return users_climbs.fillna(0)


test = get_users_climbs(path_to_user_ids='user_ids/ExampleID.csv')
