""" 
This method changes the .JSON document to make it more readable
"""

import json


def to_json(filename: str, indent=4):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=indent)

path = "LP_model.json"

to_json(path)