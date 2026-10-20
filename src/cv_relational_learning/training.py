"""
Training and evaluation loop shared by all experiments.

Behaviour that the paper results depend on (kept deliberately):
    - batches are NOT shuffled; with batch_size equal to the number of training points per class,
      every batch therefore contains a single class (datasets list class 1 first, then class 0);
    - the model is saved at every evaluation epoch, so the stored model is the one from the last
      evaluation epoch (the final epoch when n_epochs is a multiple of eval_interval);
    - the loop runs n_epochs + 1 times (epochs 0..n_epochs).
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, n_epochs, eval_interval, save_dir, verbose=True):
        self.n_epochs = n_epochs
        self.eval_interval = eval_interval
        self.save_dir = Path(save_dir)
        self.verbose = verbose
        self.train_losses = np.full(n_epochs + 1, np.nan)
        self.val_losses = np.full(n_epochs // eval_interval + 1, np.nan)

    def train(self, model, train_data, val_data, batch_size, optimiser, criterion):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        train_loader = DataLoader(train_data, batch_size=batch_size)
        val_loader = DataLoader(val_data, batch_size=batch_size) if val_data is not None else None

        model.train()
        for epoch in range(self.n_epochs + 1):
            running_loss = 0.0
            for batch_data, batch_targets in train_loader:
                loss = criterion(model(batch_data), batch_targets)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                running_loss += loss.item() * batch_targets.size(0)
            self.train_losses[epoch] = running_loss / len(train_loader.dataset)

            if epoch % self.eval_interval == 0:
                if val_loader is not None:
                    self.val_losses[epoch // self.eval_interval] = self.validate(model, val_loader, criterion)
                if self.verbose:
                    print(f"epoch {epoch:4d}  train loss {self.train_losses[epoch]:.4f}  "
                          f"val loss {self.val_losses[epoch // self.eval_interval]:.4f}")
                self.save(model)

    def validate(self, model, val_loader, criterion):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_targets in val_loader:
                running_loss += criterion(model(batch_data), batch_targets).item() * batch_targets.size(0)
        model.train()
        return running_loss / len(val_loader.dataset)

    def save(self, model):
        torch.save(model.state_dict(), self.save_dir / "model_state.pth")
        np.save(self.save_dir / "train_losses.npy", self.train_losses)
        np.save(self.save_dir / "val_losses.npy", self.val_losses)


def train_model(model, train_data, val_data, n_epochs, eval_interval, lr, batch_size, save_dir, verbose=True):
    """Adam + BCEWithLogitsLoss, as used for every model in the paper."""
    trainer = Trainer(n_epochs, eval_interval, save_dir, verbose)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    trainer.train(model, train_data, val_data, batch_size, optimiser, nn.BCEWithLogitsLoss())
    return trainer


def load_model(model, save_dir):
    model.load_state_dict(torch.load(Path(save_dir) / "model_state.pth"))
    model.eval()
    return model


def predict(model, dataset):
    """Model outputs on a whole dataset (one batch, no gradients)."""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        (batch_data, _), = loader
        return model(batch_data)


def test_model(model, test_data, save_dir):
    """Reload the saved model, predict on the test set and store preds/targets next to it."""
    model = load_model(model, save_dir)
    preds = predict(model, test_data).numpy()
    targets = test_data.targets.numpy()
    np.save(Path(save_dir) / "preds.npy", preds)
    np.save(Path(save_dir) / "targets.npy", targets)
    return preds, targets
