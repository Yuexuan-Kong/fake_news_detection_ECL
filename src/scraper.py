import json
import os

import requests
from dotenv import load_dotenv

from utils import list_to_full_string

load_dotenv()

headers = {
    'Authorization': f"Bearer {os.environ['BEARER_TOKEN']}"
    }

def get_tweets(ids:list):

    ids_str = list_to_full_string(ids)

    url = f"https://api.twitter.com/2/tweets?ids={ids_str}"

    response = requests.request("GET", url, headers=headers, data={})

    res = response.json()['data']
    return res


