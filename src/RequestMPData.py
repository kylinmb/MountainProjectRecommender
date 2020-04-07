"""
This file contains functions to connect to all the mountain project
end point. The mountain project API is explained here:
https://www.mountainproject.com/data/
"""
import json
import requests
from requests.exceptions import RequestException

base_url = 'https://www.mountainproject.com/data/'
key = '<SET YOUR KEY HERE>'


def get_user(user_id):
    """
    Requests user data from mountain project
    :param user_id: the user's id
    :return: dictionary of users information
    """
    full_url = base_url + 'get-user?userId=' + user_id + '&key=' + key
    response = requests.get(full_url)
    if response.status_code != 200:
        raise RequestException('Get User failed with status code: {}'.format(response.status_code))
    return json.loads(response.text)


def get_ticks(user_id):
    """
    Requests user's ticks from mountain project
    :param user_id:  The user's id
    :return: dictionary of user's ticks
    """
    full_url = base_url + 'get-ticks?userId=' + str(user_id) + '&key=' + key
    response = requests.get(full_url)
    if response.status_code != 200:
        raise RequestException('Get Ticks failed with status code: {}'.format(response.status_code))
    return json.loads(response.text)


def get_todos(user_id):
    """
    Returns user's to do's
    :param user_id: The user's id
    :return: dictionary of user's to do's
    """
    full_url = base_url + 'get-to-dos?userId=' + user_id + '&key=' + key
    response = requests.get(full_url)
    if response.status_code != 200:
        raise RequestException('Get To Dos failed with status code: {}'.format(response.status_code))
    return json.loads(response.text)


def get_routes(route_ids):
    """
    Returns information on routes
    :param route_ids: List of route id's
    :return: dictionary of route information
    """
    route_ids = ','.join(str(r) for r in route_ids)
    full_url = base_url + 'get-routes?routeIds=' + route_ids + '&key=' + key
    response = requests.get(full_url)
    if response.status_code != 200:
        raise RequestException('Get To Dos failed with status code: {}'.format(response.status_code))
    return json.loads(response.text)
