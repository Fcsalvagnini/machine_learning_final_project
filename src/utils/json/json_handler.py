import json

from typing import Dict


class JsonHandler:

    @staticmethod
    def parse_json_to_dict(json_path: str) -> Dict:
        with open(json_path, 'r') as j:
            parsed_json = json.load(j)
        return parsed_json