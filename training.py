import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb

from unet import UNet

# Function to calculate loss (replace with your desired loss function)
def dice_loss(pred, target):
    smooth = 1.
    intersection = (pred * target).sum(dim=(2,3)) 
    loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth))
    return loss.mean()

def train_model(model, train_loader, val_loader, epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = dice_loss 

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  

            if i % 10 == 9: 
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.3f}')
                wandb.log({"train_loss": running_loss / 10})
                running_loss = 0.0

        # Validation (optional)
        if val_loader:
            with torch.no_grad():
                val_loss = 0.0
                for _, data in enumerate(val_loader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                print(f'Epoch [{epoch+1}], Val Loss: {val_loss / len(val_loader):.3f}')
                wandb.log({"val_loss": val_loss / len(val_loader)})

if __name__ == "__main__":
    wandb.init(project="unet-segmentation") 

    config = {
        "epochs": 10,
        "learning_rate": 0.001,
        "batch_size": 4,  
    }
    wandb.config.update(config)

    # Load your training and validation datasets
    train_loader = DataLoader(...) 
    val_loader = DataLoader(...) 

    model = UNet(in_channels=6, out_channels=1) 
    train_model(model, train_loader, val_loader, config["epochs"], config["learning_rate"])

    wandb.save('model.pth') 
