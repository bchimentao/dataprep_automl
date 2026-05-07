import yaml
from experiments.experiment_runner import ExperimentRunner

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config("../configs/experiment.yaml")

    runner = ExperimentRunner(config)
    runner.run()