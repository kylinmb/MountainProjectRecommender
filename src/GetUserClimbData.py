from src import RequestMPData as mp
import pandas as pd


def get_users_climbs(path_to_user_ids):
    """
    Creates data frame where rows are users and columns are climbs.
    Entries represent a user's rating for a climb.
    If a user has completed a climb, but not given it a rating the minimum rating is assumed.
    If a user has completed a climb and given it a rating their rating is used.
    A zero indicates that a user has not completed that climb
    :param path_to_user_ids: path to list of user_ids
    :return: DataFrame users x climbs
    """
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


def get_users_climbs_binary(path_to_user_ids):
    """
    Creates data frame where rows are users and columns are climbs.
    Entries represent if a user has completed a climb.
    A one indicates that a user has completed that climb.
    A zero indicates that a user has not completed that climb
    :param path_to_user_ids: path to list of user_ids
    :return: DataFrame users x climbs
    """
    user_ids = pd.read_csv(path_to_user_ids, sep='\n', header=None)
    users_climbs = pd.DataFrame()
    for index, user_id in user_ids.iterrows():
        single_user_climb = pd.DataFrame()
        single_user_climb['user_id'] = user_id
        users_ticks = mp.get_ticks(user_id[0])['ticks']
        for t in users_ticks:
            route_id = t['routeId']
            single_user_climb[route_id] = 1
        users_climbs = pd.concat([users_climbs, single_user_climb], sort=False, ignore_index=True)
    return users_climbs.fillna(0)

