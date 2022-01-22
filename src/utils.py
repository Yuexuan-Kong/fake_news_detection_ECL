import json

def list_to_full_string(input_list:list) -> str:
    s = json.dumps(input_list).replace('[','').replace(']','')
    s = s.replace(', ',',')
    s = s.replace('"','')
    s = s.replace('\'','')
    return s