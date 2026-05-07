from .base import BalancingStrategy

class NoBalancing(BalancingStrategy):
    def apply(self, X, y):
        return X, y