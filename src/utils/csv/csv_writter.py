import csv
from typing import List

class CsvWritter:
    def __init__(self, path:str, header:List) -> None:
        self.path = path
        with open(self.path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(header)

    def write_line(self, content):
        with open(self.path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(content)