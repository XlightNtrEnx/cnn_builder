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
            nn.Conv2d(3, 32, kernel_size=3, stride=1,
                      padding="same"),  # 224 x 224 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 x 112 x 32
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1,
                      padding="same"),  # 112 x 112 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 x 56 x 64
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1,
                      padding="same"),  # 56 x 56 x 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 x 28 x 128
            nn.Dropout(0.1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1,
                      padding="same"),  # 28 x 28 x 256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14 x 14 x 256
            nn.Dropout(0.1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1,
                      padding="same"),  # 14 x 14 x 512
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7 x 7 x 512
            nn.Dropout(0.1),
        )
        self.classifer = nn.Sequential(
            nn.Linear(7 * 7 * 512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer

            nn.Linear(256, 1)  # Output layer
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
        return optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, dampening=0.1, weight_decay=1e-4)

    def _create_dataset(self):
        cat_dataset = KagglehubDataset(
            kagglehub_url="crawford/cat-dataset",
            label=0,
            max_size=512)
        dog_dataset = KagglehubDataset(
            kagglehub_url="jessicali9530/stanford-dogs-dataset",
            label=1,
            max_size=512)
        dataset = ConcatDataset(
            [cat_dataset, dog_dataset])
        return dataset

    def _create_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
        return train_transform, val_transform

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

        # Compute accuracy
        accuracy = correct_predictions / total_samples

        # Summary
        summary = {"loss": total_loss, "avg_loss_per_batch": total_loss /
                   total_samples, "accuracy": accuracy}

        return summary, inputs, labels

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

                # Compute the loss
                vloss = loss_fn(voutputs, vlabels)
                total_vloss += vloss.item()
                total_samples += vlabels.size(0)

                # Compute accuracy
                predicted = (torch.sigmoid(voutputs) > 0.5).float()
                correct_predictions += (predicted == vlabels).sum().item()
        accuracy = correct_predictions / total_samples

        summary = {"loss": total_vloss, "avg_loss_per_batch": total_vloss /
                   total_samples, "accuracy": accuracy}

        return summary, vinputs, vlabels

    def _get_target_layer(self):
        return self.features[-1]
