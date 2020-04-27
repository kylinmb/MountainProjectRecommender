import math
import numpy as np
import pandas as pd
from random import sample
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


"""
This file contains the data to calculate and evaluates
user based rating predictions for routes
"""


def pairwise_pearson_correlation(data_matrix):
    """
    Computes the pearson coefficient between every pair of users
    :param data_matrix: users X ratings - numpy array, do not include the user_id
    :return: users X users matrix of pearson coefficient
    """
    size = data_matrix.shape[0]
    correlations = np.zeros((size, size))
    for i, row_i in enumerate(data_matrix):
        for j, row_j in enumerate(data_matrix):
            correlations[i][j], _ = pearsonr(row_i, row_j)
    return correlations


def find_k_similar_neighbors(k, similarity_matrix):
    """
    Find the k nearest neighbors for each user based
    on values in similarity matrix
    :param k: number of nearest neighbors to find
    :param similarity_matrix: pairwise measures of similarity
    :return: list of k nearest neighbors for each user
    """
    neighbors = []
    for i, row in enumerate(similarity_matrix):
        row_nn = np.argpartition(row, -(k + 1))[-(k + 1):]
        row_nn = np.delete(row_nn, np.where(row_nn == i))
        neighbors.append(row_nn)
    return neighbors


def predicted_rating(climb, user, ratings, nearest_neighbors, similarities):
    """
    This method predict the rating for a climb given a user and the complete table
    of user x climb ratings.
    :param climb: Climb to get rating for
    :param user: User to rate climb for
    :param ratings: Complete table of user x climb ratings
    :param nearest_neighbors: How many nearest neighbors to use in rating prediction calculation
    :param similarities: Table of user x user similarity measures
    :return:
    """
    numerator = 0.0
    denominator = 0.0
    nearest_neighbor = nearest_neighbors[user]
    for nn in nearest_neighbor:
        nn_rating = ratings.iloc[nn].loc[climb]
        if nn_rating != 0:
            nn_mean = ratings.iloc[nn].replace(0, np.NaN).mean()
            numerator += similarities[user][nn] * (nn_rating - nn_mean)
            denominator += similarities[user][nn]
    user_mean = ratings.iloc[user].replace(0, np.NaN).mean()
    # denominator will be zero if none of their nearest neighbors have rated that climb
    return 0.0 if denominator == 0 else user_mean + numerator / denominator


def hold_out_evaluation(list_of_users, ground_truth_ranking, percent_to_hold_out, k, similarity=pairwise_pearson_correlation):
    """
    This method will hold out a percent of the climbs for each of the users listed and
    predict the rating for each of the held out climbs and compare those to the known
    rating for that user.
    :param list_of_users: users to hold out climbs for
    :param ground_truth_ranking: the ground truth rankings of all users and all climbs
    :param percent_to_hold_out: percentage of climbs to hold out as decimal
    :param k: number of nearest neighbors to include in rating prediction
    :return: Mean Absolute Error and Root Mean Squared Error for each user
    """
    hold_out_rating = ground_truth_ranking.copy()
    all_user_hold_out_climbs = []
    # randomly select set of climbs to hold out
    for u in list_of_users:
        single_user_ratings = ground_truth_ranking.iloc[u]
        climbed_routes = single_user_ratings.loc[single_user_ratings.values > 0].keys()
        hold_out_num = math.ceil(len(climbed_routes) * percent_to_hold_out)
        hold_out_climbs = sample(list(climbed_routes), hold_out_num)
        all_user_hold_out_climbs.append({'user_id': u, 'climbs': hold_out_climbs})
        for c in hold_out_climbs:
            hold_out_rating.iloc[u].loc[c] = 0

    # now calculate predictions using hold out set
    sim = similarity(hold_out_rating.to_numpy())
    nearest_neighbors = find_k_similar_neighbors(k, sim)

    # now for each user calculate ap
    mean_squared_errors = []
    mean_absolute_errors = []
    for user in all_user_hold_out_climbs:
        predictions = []
        ground_truths = []
        for climb in user['climbs']:
            predictions.append(predicted_rating(climb, user['user_id'], hold_out_rating, nearest_neighbors, sim))
            ground_truths.append(ground_truth_ranking.iloc[user['user_id']].loc[climb])
        mean_squared_errors.append(mean_squared_error(ground_truths, predictions))
        mean_absolute_errors.append(mean_absolute_error(ground_truths, predictions))
    return mean_absolute_errors, np.sqrt(mean_squared_errors)


def eval_five_fold(k, rating_df, similarity=pairwise_pearson_correlation):
    """
    Evaluate five folds of hold out data
    :param k: number of nearest neighbors
    :param rating_df: complete rating data frame
    :return:
    """
    folds = []
    user_indices = [*range(0, 30)]
    for i in range(0, 5):
        folds.append(sample(user_indices, 6))
        user_indices = [x for x in user_indices if x not in folds[i]]

    maes = []
    mses = []
    for fold in folds:
        mae, mse = hold_out_evaluation(fold, rating_df, 0.20, k, similarity)
        maes.append(np.mean(mae))
        mses.append(np.mean(mse))
    return {'mean': np.mean(maes), 'std': np.std(maes) }, {'mean': np.mean(mses), 'std': np.std(mses)}


def get_ranking_data(list_of_users, ground_truth_ranking, percent_to_hold_out, k, similarity=pairwise_pearson_correlation):
    """
    This method will hold out a percent of the climbs for each of the users listed and
    predict the rating for each of the held out climbs and compare those to the known
    rating for that user.
    :param list_of_users: users to hold out climbs for
    :param ground_truth_ranking: the ground truth rankings of all users and all climbs
    :param percent_to_hold_out: percentage of climbs to hold out as decimal
    :param k: number of nearest neighbors to include in rating prediction
    :return: Mean Absolute Error and Root Mean Squared Error for each user
    """
    hold_out_rating = ground_truth_ranking.copy()
    all_user_hold_out_climbs = []
    # randomly select set of climbs to hold out
    for u in list_of_users:
        single_user_ratings = ground_truth_ranking.iloc[u]
        climbed_routes = single_user_ratings.loc[single_user_ratings.values > 0].keys()
        hold_out_num = math.ceil(len(climbed_routes) * percent_to_hold_out)
        hold_out_climbs = sample(list(climbed_routes), hold_out_num)
        all_user_hold_out_climbs.append({'user_id': u, 'climbs': hold_out_climbs})
        for c in hold_out_climbs:
            hold_out_rating.iloc[u].loc[c] = 0

    # now calculate predictions using hold out set
    sim = similarity(hold_out_rating.to_numpy())
    nearest_neighbors = find_k_similar_neighbors(k, sim)

    # now for each user get data calculate ap
    user_data = []
    for user in all_user_hold_out_climbs:
        single_user_data = {'user_index': user['user_id'], 'climbs': []}
        for climb in user['climbs']:
            climb_data = {'route_id': climb,
                          'ground_truth': ground_truth_ranking.iloc[user['user_id']].loc[climb],
                          'predicted': predicted_rating(climb, user['user_id'], hold_out_rating, nearest_neighbors, sim)}
            single_user_data['climbs'].append(climb_data)
        user_data.append(single_user_data)
    return user_data


def create_galago_data(rating_dictionary, rating_df_with_id, number_of_rel_docs):
    # Create Judgments - These are the known values
    # Create Baseline - These are the predicted files
    judgments = pd.DataFrame(columns=['qid', '0', 'docid', 'rel'])
    baseline = pd.DataFrame(columns=['qid', '0', 'docid', 'rank', 'score'])
    for rating in rating_dictionary:
        qid = rating_df_with_id.iloc[rating['user_index']].loc['user_id']
        for climb in rating['climbs']:
            judgments = judgments.append({'qid': qid,
                                          '0': '0',
                                          'docid': climb['route_id'],
                                          'rel': climb['ground_truth']}, ignore_index=True)
            baseline = baseline.append({'qid': qid,
                                        '0': '0',
                                        'docid': climb['route_id'],
                                        'rank': 0,
                                        'score': climb['predicted']}, ignore_index=True)
    # set rank based on rating
    for i in baseline['qid'].unique():
        temp = baseline[baseline['qid'] == i]
        temp = temp.sort_values(by='score', ascending=False)
        count = temp['qid'].count() + 1
        temp['rank'] = [k for k in range(1, count)]
        baseline.update(temp)
    judgments = judgments.astype({'qid': int, 'docid': int})
    baseline = baseline.astype({'qid': int, 'docid': int, 'rank': int})
    baseline = baseline[baseline['rank'] < number_of_rel_docs]
    return judgments, baseline


def eval_five_fold_ranking(k, rating_df, rating_df_with_id, judgments_path, baseline_path, number_of_rel_docs, similarity=pairwise_pearson_correlation):
    """
    Evaluate five folds of hold out data
    :param k: number of nearest neighbors
    :param rating_df: complete rating data frame
    :return:
    """
    folds = []
    user_indices = [*range(0, 30)]
    for i in range(0, 5):
        folds.append(sample(user_indices, 6))
        user_indices = [x for x in user_indices if x not in folds[i]]

    judgments = pd.DataFrame()
    baseline = pd.DataFrame()
    for fold_num, fold in enumerate(folds):
        ranking_dictionary = get_ranking_data(fold, rating_df, 0.20, k, similarity)
        fold_judgments, fold_baseline = create_galago_data(ranking_dictionary, rating_df_with_id, number_of_rel_docs)
        fold_judgments.to_csv(judgments_path + str(fold_num) + '.txt', sep=' ', header=False, index=False)
        fold_baseline.to_csv(baseline_path + str(fold_num) + '.txt', sep=' ', header=False, index=False)
        judgments = pd.concat([judgments, fold_judgments], sort=False, ignore_index=True)
        baseline = pd.concat([baseline, fold_baseline], sort=False, ignore_index=True)
    judgments.to_csv(judgments_path + '.txt', sep=' ', header=False, index=False)
    baseline.to_csv(baseline_path + '.txt', sep=' ', header=False, index=False)


