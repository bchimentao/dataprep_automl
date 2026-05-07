import json

class ResultWriter:
    def save(self, results, path):
        with open(path, "w") as f:
            json.dump(results, f, indent=4)