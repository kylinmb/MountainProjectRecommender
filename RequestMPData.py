import json
import requests
from requests.exceptions import RequestException

base_url = 'https://www.mountainproject.com/data/'
key = '<SET YOUR KEY HERE>'


def get_user(user_id):
    full_url = base_url + 'get-user?userId=' + user_id + '&key=' + key
    response = requests.get(full_url)
    if response.status_code != 200:
        raise RequestException('Get User failed with status code: {}'.format(response.status_code))
    return json.loads(response.text)


def get_ticks(user_id):
    full_url = base_url + 'get-ticks?userId=' + str(user_id) + '&key=' + key
    response = requests.get(full_url)
    if response.status_code != 200:
        raise RequestException('Get Ticks failed with status code: {}'.format(response.status_code))
    return json.loads(response.text)


def get_todos(user_id):
    full_url = base_url + 'get-to-dos?userId=' + user_id + '&key=' + key
    response = requests.get(full_url)
    if response.status_code != 200:
        raise RequestException('Get To Dos failed with status code: {}'.format(response.status_code))
    return json.loads(response.text)


def get_routes(route_ids):
    route_ids = ','.join(route_ids)
    full_url = base_url + 'get-routes?routeIds=' + route_ids + '&key=' + key
    response = requests.get(full_url)
    if response.status_code != 200:
        raise RequestException('Get To Dos failed with status code: {}'.format(response.status_code))
    return json.loads(response.text)
