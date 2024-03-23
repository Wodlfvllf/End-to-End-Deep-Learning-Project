import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import copy
from tqdm import tqdm
import numpy as np

class CustomTrainer:
    def __init__(self, model, train_loader, val_loader, lr=1.5e-5, criterion=nn.BCELoss(), device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.criterion = criterion
        self.device = device

    def train(self, epochs, save_path):
        best_acc = -np.inf
        best_weights = None
        accuracy = Accuracy(task='binary').to(self.device)
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            train_pred = []
            val_pred = []

            # Training Loop
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{epochs} - Training'):
                images, labels = batch['img'].to(self.device), batch['label'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_pred.append(loss.item())
                train_acc = accuracy(outputs, labels)
                train_accuracies.append(train_acc.item())

            train_loss = np.mean(train_pred)

            # Validation Loop
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc=f'Epoch {epoch + 1}/{epochs} - Validation'):
                    images, labels = batch['img'].to(self.device), batch['label'].to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_pred.append(loss.item())
                    val_acc = accuracy(outputs, labels)
                    val_accuracies.append(val_acc.item())

            val_loss = np.mean(val_pred)

            # Print and store losses and accuracies
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {np.mean(train_accuracies):.4f}, Valid Loss: {val_loss:.4f}, Valid Accuracy: {np.mean(val_accuracies):.4f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Save best model
            if max(train_accuracies) > best_acc:
                best_acc = max(train_accuracies)
                best_weights = copy.deepcopy(self.model.state_dict())

        # Save the best model
        torch.save(best_weights, save_path)

        return train_losses, val_losses, train_accuracies, val_accuracies
