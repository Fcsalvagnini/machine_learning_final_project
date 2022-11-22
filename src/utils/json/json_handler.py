import json

from pathlib import Path
import os
from typing import Dict

def parse_json_to_dict(data_descriptors_path:str, phase:str) -> Dict:
    root_dir = os.path.join(Path.cwd(), Path(data_descriptors_path))
    json_path = os.path.join(root_dir, Path(f"{phase}.json"))
    with open(json_path, 'r') as j:
        parsed_json = json.load(j)
    return parsed_json