import json

from pathlib import Path
from typing import Dict


class JsonHandler:

    @staticmethod
    def parse_json_to_dict(phase: str) -> Dict:
        root_dir = Path.cwd() /  Path("src/data/descriptors")
        json_path = str(root_dir / Path(f"{phase}.json"))
        with open(json_path, 'r') as j:
            parsed_json = json.load(j)
        return parsed_json