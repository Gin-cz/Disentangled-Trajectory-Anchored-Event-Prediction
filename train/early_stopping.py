"""Track validation progress for early stopping."""

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    def step(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
            return False
        elif val_metric < self.best_score - self.delta:
            self.best_score = val_metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop
