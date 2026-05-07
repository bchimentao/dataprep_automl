from data.dataset_loader import DatasetLoader
from balancing.no_balancing import NoBalancing
from automl.autogluon_runner import AutoGluonRunner
from evaluation.evaluator import Evaluator

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.loader = DatasetLoader()
        self.evaluator = Evaluator()

    def run(self):
        for dataset_config in self.config["datasets"]:
            X, y = self.loader.load(
                dataset_config["path"],
                dataset_config["target"]
            )

            for balancing_name in self.config["balancing_strategies"]:
                balancing = self.get_balancing(balancing_name)
                X_bal, y_bal = balancing.apply(X, y)

                for framework_name in self.config["frameworks"]:
                    model = self.get_framework(framework_name)

                    model.train(X_bal, y_bal)
                    y_pred = model.predict(X_bal)

                    results = self.evaluator.evaluate(y_bal, y_pred)
                    print(results)

    def get_balancing(self, name):
        if name == "none":
            return NoBalancing()
        else:
            raise ValueError("Balanceamento não implementado")

    def get_framework(self, name):
        if name == "autogluon":
            return AutoGluonRunner()
        else:
            raise ValueError("Framework não implementado")