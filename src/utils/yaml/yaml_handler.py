import yaml

from pathlib import Path
from typing import Dict


class YamlHandler:

    @staticmethod
    def parse_yaml_to_dict(config: str) -> Dict:
        root_dir = Path.cwd() /  Path("src/experiment_configs")
        yaml_path = str(root_dir / Path(f"{config}"))
        with open(yaml_path, 'r') as y:
            parsed_yaml = yaml.load(y, Loader=yaml.FullLoader)
        return parsed_yaml