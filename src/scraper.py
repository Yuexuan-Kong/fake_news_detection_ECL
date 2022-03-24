import json
from datetime import datetime, date
import os

import requests
from dotenv import load_dotenv
import numpy as np
from soupsieve import escape
from tqdm import tqdm

from utils import list_to_full_string

load_dotenv()

headers = {
    'Authorization': f"Bearer {os.environ['BEARER_TOKEN']}"
    }

def get_tweets(input_ids):

    n = len(input_ids)

    if n > 99:

        n_sections = int(round(n / 99))
        split_ids = np.array_split(np.array(input_ids),n_sections)

    else:
        n_sections = 1
        split_ids = [input_ids]

    full_res = []

    for ids in tqdm(split_ids):

        try:

            ids_str = list_to_full_string(ids)

            url = f"https://api.twitter.com/2/tweets?ids={ids_str}"

            response = requests.request("GET", url, headers=headers, data={})

            res = response.json()['data']

            del response

            full_res += res

            del res

        except:

            pass
        
    return full_res

def get_today_tweets(research_query,max_results):

    start_time = str(date.today()) + "T00:00:00.00Z"

    url = f"https://api.twitter.com/2/tweets/search/recent?query=({research_query}) lang:en -is:retweet -is:reply -is:quote&start_time={start_time}&max_results={max_results}&tweet.fields=created_at"

    payload={}

    response = requests.request("GET", url, headers=headers, data=payload)

    try:

        res = response.json()['data']
    
    except:
        
        res = []

    return res


