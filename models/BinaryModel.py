import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.optim as optim

from dataset import KagglehubDataset, TransformTorchDataset

from . import AbstractModel


class BinaryModel(AbstractModel):
    def __init__(self):
        super(BinaryModel, self).__init__(name="CatDogClassifier")

    def _init_layers(self):
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding="same"),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="same"),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
        )
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifer = nn.Sequential(
            nn.Linear(256 * 28 * 28, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1)  # Output layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        return x

    def _create_loss_fn(self):
        return nn.BCEWithLogitsLoss()

    def _create_optimizer(self):
        return optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, dampening=0.1)

    def _create_loaders(self):
        # Datasets
        cat_dataset = KagglehubDataset(
            kagglehub_url="crawford/cat-dataset",
            label=0,
            max_size=512)
        dog_dataset = KagglehubDataset(
            kagglehub_url="jessicali9530/stanford-dogs-dataset",
            label=1,
            max_size=512)

        # Merge datasets and split into training and validation
        combined_dataset = ConcatDataset(
            [cat_dataset, dog_dataset])
        train_size = 0.7
        val_size = 0.3
        train_dataset, val_dataset = torch.utils.data.random_split(
            combined_dataset, [train_size, val_size])

        # Assign transformers to datasets
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        train_dataset = TransformTorchDataset(
            train_dataset, transform=train_transform)
        val_dataset = TransformTorchDataset(
            val_dataset, transform=val_transform)

        print(f'Training set has {len(train_dataset)} instances')
        print(f'Validation set has {len(val_dataset)} instances')

        # Loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        return train_loader, val_loader

    def _train_one_epoch(self, train_loader: DataLoader, loss_fn: nn.BCEWithLogitsLoss,
                         optimizer: torch.optim.Optimizer, device: torch.device):
        total_loss = 0.
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            # Assign data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # If using BCEWithLogitsLoss, reshape labels
            labels = labels.view(-1, 1).float()

            # Make predictions for this batch
            outputs = self(inputs)

            # Compute the batch's loss
            loss = loss_fn(outputs, labels)

            # Zero, backpropagate, and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

            # Compute total samples
            total_samples += labels.size(0)

            # Compute correct predictions
            with torch.no_grad():
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_predictions += (predicted == labels).sum().item()

        # Compute average loss and accuracy
        avg_batch_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples

        # Summary
        summary = {"avg_batch_loss": avg_batch_loss,
                   "total_loss": total_loss, "accuracy": accuracy}

        return summary

    def _validate_one_epoch(self, val_loader: DataLoader, loss_fn: nn.BCEWithLogitsLoss, device: torch.device):
        total_vloss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for vinputs, vlabels in val_loader:
                # Assign data to device
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)

                # If using BCEWithLogitsLoss, reshape labels
                vlabels = vlabels.view(-1, 1).float()

                # Make predictions for this batch
                voutputs = self(vinputs)

                # Compute the loss and accuracy
                vloss = loss_fn(voutputs, vlabels)
                total_vloss += vloss.item()
                total_samples += vlabels.size(0)
                predicted = (torch.sigmoid(voutputs) > 0.5).float()
                correct_predictions += (predicted == vlabels).sum().item()
        avg_batch_vloss = total_vloss / len(val_loader)
        accuracy = correct_predictions / total_samples

        summary = {"avg_batch_loss": avg_batch_vloss,
                   "total_loss": total_vloss, "accuracy": accuracy}

        return summary, vinputs, vlabels

    def _get_target_layer(self):
        return self.features[-1]