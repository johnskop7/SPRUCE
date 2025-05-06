import os
import torch

class EarlyStopping:
    """
    Stops training if monitored metric does not improve for a given patience.
    Maintains best_score and num_bad_epochs.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.num_bad_epochs = 0

    def __call__(self, current_score: float) -> bool:
        """
        Call after each validation. Returns True if training should stop.
        """
        if (self.best_score is None) or (current_score > self.best_score + self.min_delta):
            self.best_score = current_score
            self.num_bad_epochs = 0
            return False
        else:
            self.num_bad_epochs += 1
            return self.num_bad_epochs >= self.patience

    def __str__(self):
        return (f"EarlyStopping(patience={self.patience}, "
                f"min_delta={self.min_delta}, "
                f"best_score={self.best_score}, "
                f"num_bad_epochs={self.num_bad_epochs})")

class ModelCheckpoint:
    """
    Saves model when monitored metric improves. Always overwrites a single file.
    Attributes:
        dirpath (str): directory to save the model
        filename (str): name of the checkpoint file
        best_score (float): best observed score
        last_path (str): full path to saved checkpoint
    """
    def __init__(self, dirpath: str, filename: str = 'best_model.pth', mode: str = 'max'):
        os.makedirs(dirpath, exist_ok=True)
        self.dirpath = dirpath
        self.filename = filename
        self.mode = mode
        self.best_score = None
        self.last_path = None
        if mode == 'max':
            self._compare = lambda a, b: a > b
        else:
            self._compare = lambda a, b: a < b

    def __call__(self, model: torch.nn.Module, epoch: int, current_score: float):
        """
        Save model.state_dict() if current_score improves over best_score.
        Always writes to the same file, overwriting previous.
        Accepts an epoch argument for compatibility but does not use it in naming.
        """
        if (self.best_score is None) or self._compare(current_score, self.best_score):
            self.best_score = current_score
            path = os.path.join(self.dirpath, self.filename)
            torch.save(model.state_dict(), path)
            self.last_path = path

    def load_best(self, model: torch.nn.Module) -> None:
        """
        Loads the best checkpoint into `model`.
        Raises if no checkpoint was saved.
        """
        if self.last_path is None:
            raise ValueError("No checkpoint has been saved yet.")
        state = torch.load(self.last_path)
        model.load_state_dict(state)
        return model

    def __str__(self):
        return (f"ModelCheckpoint(dirpath={self.dirpath}, filename={self.filename}, "
                f"best_score={self.best_score})")
