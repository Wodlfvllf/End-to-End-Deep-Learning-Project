from utils_c1 import *
from Model_c1 import *
from dataset_c1 import *

class Trainer():
    def __init__(self, fold, model, train_dataloader, test_dataloader):
        self.fold = fold
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def model_train(self, fold, epochs):
        # Loss function and optimizer
        DEVICE = torch.device("cuda")
        criterion = nn.BCELoss()  # Binary Cross Entropy
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
    #     optimizer = optimizer.to(DEVICE)
        accuracy = Accuracy(task = 'binary').to(DEVICE)
    
        # Hold the best model
        best_acc = -np.inf  # Init to negative infinity
        best_weights = None

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in tqdm(range(epochs), desc='Epochs', unit='epoch'):
            train_pred = []
            val_pred = []

            self.model.train()
            for batch in self.train_dataloader:
                images, labels = batch['image'], batch['labels']
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_pred.append(loss.item())

                # Calculate training accuracy
                train_acc = accuracy(outputs, labels)
                train_accuracies.append(train_acc.item())

            train_loss = np.mean(train_pred)

            self.model.eval()
            with torch.no_grad():
                for val_batch in self.test_dataloader:
                    val_images, val_labels = val_batch['image'], val_batch['labels']
                    val_images = val_images.to(DEVICE)
                    val_labels = val_labels.to(DEVICE)

                    val_outputs = self.model(val_images)
                    val_loss = criterion(val_outputs, val_labels)
                    val_pred.append(val_loss.item())

                    # Calculate validation accuracy
                    val_acc = accuracy(val_outputs, val_labels)
                    val_accuracies.append(val_acc.item())

            val_loss = np.mean(val_pred)

            # Print and store losses and accuracies
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {np.mean(train_accuracies):.4f}, Valid Loss: {val_loss:.4f}, Valid Accuracy: {np.mean(val_accuracies):.4f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Save best model
            if max(train_accuracies) > best_acc:
                best_acc = max(train_accuracies)
                best_weights = copy.deepcopy(self.model.state_dict())
                
    #         model.load_state_dict(best_weights)
        
        # Save the best model
        torch.save(best_weights, f'./best_model_{fold}.pth')

        # Plot training and validation losses
        return train_losses, val_losses, train_accuracies, val_accuracies
