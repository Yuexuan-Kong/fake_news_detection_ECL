import json
import os

import requests
from dotenv import load_dotenv
import numpy as np
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


