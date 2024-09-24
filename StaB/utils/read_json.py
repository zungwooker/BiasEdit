import json
import os

def load_json(json_path: str):
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            try:
                json_file = json.load(file)
            except json.JSONDecodeError:
                raise RuntimeError("An error occurred while loading the existing json file.")
    else:
        raise RuntimeError(f".json does not exist.\nPath: {json_path}")
    
    return json_file