# from Model_mae import *
# from utils_mae import *
# from dataset_mae import *

# class ModelTrainer:
#     def __init__(self, model, device, epochs=80, batch_size=64):
        
#         self.model = model
#         self.device = device
#         self.epochs = epochs
#         self.batch_size = batch_size

#     def model_train(self, train_dataloader, val_dataloader):

#         optimizer = AdamW(self.model.parameters(), lr=1.5e-4, weight_decay=0.05)
#         scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)

#         train_losses = []
#         val_losses = []

#         for epoch in range(self.epochs):
#             train_loss = 0.0
#             val_loss = 0.0

#             self.model.train()
#             for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{self.epochs} (Train)', unit='batch'):
#                 images = batch['img'].to(self.device).float()

#                 optimizer.zero_grad()
#                 imgs, outputs, ind = self.model(images)
#                 loss = custom_loss(imgs, outputs, ind)
#                 loss.backward()
#                 optimizer.step()

#                 train_loss += loss.item()

#             train_loss /= len(train_dataloader)
#             train_losses.append(train_loss)

#             self.model.eval()
#             with torch.no_grad():
#                 for batch in tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{self.epochs} (Val)', unit='batch'):
#                     images = batch['img'].to(self.device).float()

#                     imgs, val_outputs, ind = self.model(images)
#                     loss = custom_loss(imgs, val_outputs, ind)

#                     val_loss += loss.item()

#             val_loss /= len(val_dataloader)
#             val_losses.append(val_loss)

#             scheduler.step()

#             print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
#             with open('losses.txt', 'a') as f:  # Open file in append mode
#                 f.write(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n')

#             torch.save(self.model, f'./best_model.pth')

#         return train_losses, val_losses

from Model_mae import *
from utils_mae import *
from dataset_mae import *

class Trainer:
    def __init__(self, model, device, epochs=80, batch_size=64):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

    def model_train(self, train_dataloader, val_dataloader):

        optimizer = AdamW(self.model.parameters(), lr=1.5e-4, weight_decay=0.05)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)

        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            train_loss = 0.0
            val_loss = 0.0

            self.model.train()
            for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{self.epochs} (Train)', unit='batch'):
                images = batch['img'].to(self.device).float()

                optimizer.zero_grad()
                imgs, outputs, ind = self.model(images)
                loss = custom_loss(imgs, outputs, ind)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)

            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{self.epochs} (Val)', unit='batch'):
                    images = batch['img'].to(self.device).float()

                    imgs, val_outputs, ind = self.model(images)
                    loss = custom_loss(imgs, val_outputs, ind)

                    val_loss += loss.item()

            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)

            scheduler.step()

            print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            with open('losses.txt', 'a') as f:  # Open file in append mode
                f.write(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n')

            torch.save(self.model, f'./best_model.pth')

        return train_losses, val_losses

