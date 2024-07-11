import os

import torch


class EarlyStop:
    def __init__(
        self, patience=10, verbose=False, delta=0, save_path=None, model_name=None
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path
        self.model_name = model_name if model_name else "user_withdraw_predict_model.pt"

        self.counter = 0
        self.best_score = None
        self.val_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_model(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Early Stop counter [{self.counter}/{self.patience}]")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self._save_model(val_loss, model)
            self.counter = 0

    def _save_model(self, val_loss, model):
        os.makedirs(self.save_path, exist_ok=True)
        save_path = os.path.join(self.save_path, self.model_name)
        if self.verbose:
            print(f"Val loss decreased ({self.val_loss:.6f} --> {val_loss:.6f}).")
            self.val_loss = val_loss
            print(f'Saving model to "{save_path}"')
            torch.save(model.state_dict(), save_path)
            print("Saved model.")


class UserWithdrawPredictTrainer:
    def __init__(
        self,
        model,
        train_dataset_loader,
        val_dataset_loader,
        optimizer,
        criterion,
        epochs,
        model_path,
        model_name,
        device="cpu",
    ):
        self.model_path = model_path

        self.model = model
        self.train_dataset_loader = train_dataset_loader
        self.val_dataset_loader = val_dataset_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.device = device

        self.early_stop = EarlyStop(
            patience=20, verbose=True, save_path=model_path, model_name=model_name
        )

        self.train_losses = []
        self.val_losses = []

    def train_step(self):
        self.model.train()
        train_epoch_loss = 0
        for idx, (inputs, targets) in enumerate(self.train_dataset_loader):
            self.optimizer.zero_grad()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            pred = self.model(inputs)

            loss = self.criterion(pred, targets)

            loss.backward()
            self.optimizer.step()

            train_epoch_loss += loss.item()

        return train_epoch_loss / len(self.train_dataset_loader)

    def evaluate_step(self):
        self.model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(self.val_dataset_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                pred = self.model(inputs)

                loss = self.criterion(pred, targets)

                val_epoch_loss += loss.item()

        return val_epoch_loss / len(self.val_dataset_loader)

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.train_step()
            self.train_losses.append(train_loss)

            val_loss = self.evaluate_step()
            self.val_losses.append(val_loss)

            print(f"Epoch [{epoch+1:02}/{self.epochs}]")
            print(f"Train Loss: {train_loss:.3f}")
            print(f"Val Loss: {val_loss:.3f}")

            self.early_stop(val_loss, self.model)

            if self.early_stop.early_stop:
                print("Early stopping...")
                break
            print("#" * 70)
