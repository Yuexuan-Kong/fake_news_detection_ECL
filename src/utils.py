import json

def list_to_full_string(input_list:list) -> str:
    s = list(map(lambda x:int(x),input_list))
    s = json.dumps(s).replace('[','').replace(']','')
    s = s.replace(', ',',')
    s = s.replace('"','')
    s = s.replace('\'','')
    return s