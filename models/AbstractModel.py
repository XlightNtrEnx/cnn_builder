import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import os
import numpy as np
from typing import Dict, Any
from pytorch_grad_cam import GradCAM
from torchvision.transforms import Normalize
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import pytz


class AbstractModel(ABC, nn.Module):
    def __init__(self, name: str):
        super(AbstractModel, self).__init__()
        self.name = name
        self._init_layers()
        # print("Model initialized with parameters: ",
        #       [x for x in self.modules()])
        self.device = torch.device("cuda")
        self.to(self.device)

    @abstractmethod
    def _init_layers(self):
        """
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)\n
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)\n
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)\n
        \n
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)\n
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)\n
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)\n
        \n
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)\n
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)\n
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)\n
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)\n
        \n
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)\n
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)\n
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)\n
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)\n
        \n
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)\n
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)\n
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)\n
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)\n
        \n
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)\n
        self.fc2 = nn.Linear(4096, 4096)\n
        self.fc3 = nn.Linear(4096, 1000)
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x = F.relu(self.conv1_1(x))\n
        x = F.relu(self.conv1_2(x))\n
        x = self.pool1(x)\n
        \n
        x = F.relu(self.conv2_1(x))\n
        x = F.relu(self.conv2_2(x))\n
        x = self.pool2(x)\n
        \n
        x = F.relu(self.conv3_1(x))\n
        x = F.relu(self.conv3_2(x))\n
        x = F.relu(self.conv3_3(x))\n
        x = self.pool3(x)\n
        \n
        x = F.relu(self.conv4_1(x))\n
        x = F.relu(self.conv4_2(x))\n
        x = F.relu(self.conv4_3(x))\n
        x = self.pool4(x)\n
        \n
        x = F.relu(self.conv5_1(x))\n
        x = F.relu(self.conv5_2(x))\n
        x = F.relu(self.conv5_3(x))\n
        x = self.pool5(x)\n
        \n
        x = torch.flatten(x, 1)\n
        x = F.relu(self.fc1(x))\n
        x = F.dropout(x, 0.5, self.training)\n
        x = F.relu(self.fc2(x))\n
        x = F.dropout(x, 0.5, self.training)\n
        x = self.fc3(x)\n
        return x
        """
        pass

    @abstractmethod
    def _create_loss_fn(self) -> _Loss:
        pass

    @abstractmethod
    def _create_optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def _create_loaders(self) -> tuple[DataLoader, DataLoader]:
        '''
        train_loader = DataLoader(...)\n
        val_loader = DataLoader(...)\n
        return (train_loader, val_loader)
        '''
        pass

    @abstractmethod
    def _train_one_epoch(self, train_loader: DataLoader, loss_fn: _Loss,
                         optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float | int]:
        """
        return train_summary
        """
        pass

    @abstractmethod
    def _validate_one_epoch(self, val_loader: DataLoader, loss_fn: _Loss, device: torch.device) -> tuple[Dict[str, float | int], torch.Tensor, torch.Tensor]:
        """
        vinputs should be a tensor of shape [B, C, H, W]\n
        vlabels should be a tensor of shape [B]\n

        return val_summary, vinputs, vlabels 
        """
        pass

    @abstractmethod
    def _get_target_layer(self):
        pass

    def train_and_save_model(self, epochs: int = 1000, patience: int = 20):
        if epochs < 1:
            raise ValueError("epochs must be greater than 0")

        # Get device
        device = self.device

        # Loss function and optimizer
        loss_fn = self._create_loss_fn()
        optimizer = self._create_optimizer()
        print(
            f"Training {self.name} for {epochs} epochs with loss_fn {loss_fn} and optimizer {optimizer}")

        # Loaders
        train_loader, val_loader = self._create_loaders()
        print(
            f"Train loader has {len(train_loader)} batches. Val loader has {len(val_loader)} batches")

        # Preview a batch of training images
        images, labels = next(iter(train_loader))
        print(
            f"Batch of images shape: {images.shape}, Batch of labels shape: {labels.shape}")
        img_grid = torchvision.utils.make_grid(images)
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.title("Sample Images from Train Loader")
        plt.show()

        # Preview a batch of validation images
        images, labels = next(iter(val_loader))
        print(
            f"Batch of images shape: {images.shape}, Batch of labels shape: {labels.shape}")
        img_grid = torchvision.utils.make_grid(images)
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.title("Sample Images from Val Loader")
        plt.show()

        # Writer
        sg_timezone = pytz.timezone('Asia/Singapore')
        timestamp = datetime.now(sg_timezone).strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(f'runs/{self.name}/{timestamp}')

        # Write architecture
        random_image = torch.randn(1, 3, 224, 224).to(device)
        writer.add_graph(self, random_image)
        writer.flush()

        # Create folder to save model
        saved_model_folder_path = os.path.join(
            'trained_models', self.name, timestamp)
        if not os.path.exists(saved_model_folder_path):
            os.makedirs(saved_model_folder_path)

        highest_avg_vloss = 1_000_000.
        highest_val_accuracy = 0.
        epochs_since_best = 0
        for epoch_idx in range(epochs):
            print(f"Epoch {epoch_idx + 1} of {epochs}")

            # Train
            self.train()
            train_summary = self._train_one_epoch(
                train_loader, loss_fn, optimizer, device)

            # Validate
            self.eval()
            val_summary, vinputs, vlabels = self._validate_one_epoch(
                val_loader, loss_fn, device)

            # Summarize metrices
            self._summarize(train_summary, val_summary,
                            epoch_idx, writer)

            # Write activation map
            self.eval()
            self._write_activation_map(
                writer, target_layer=self._get_target_layer(), inputs=vinputs, labels=vlabels, epoch_idx=epoch_idx)

            # Track best performance, and save the model's state
            avg_vloss = val_summary['avg_batch_loss']
            if avg_vloss < highest_avg_vloss:
                highest_avg_vloss = avg_vloss
            validation_accuracy = val_summary['accuracy']
            if validation_accuracy > highest_val_accuracy:
                highest_val_accuracy = validation_accuracy
                epochs_since_best = 0
                model_path = os.path.join(
                    saved_model_folder_path, f"{epoch_idx + 1}.pth")
                torch.save(self.state_dict(), model_path)
            else:
                epochs_since_best += 1

            # Early stopping
            if epochs_since_best >= patience:
                print(
                    f"Stopping early after {epochs_since_best} epochs without improvement")
                break

        # Close the writer
        writer.close()

    def _summarize(self, train_summary: Dict[str, Any], val_summary: Dict[str, Any], epoch_idx: int, writer: SummaryWriter):
        for key, value in train_summary.items():
            print(f"Train/{key}: {value}")
            writer.add_scalar(f"Train/{key}", value, epoch_idx + 1)
        for key, value in val_summary.items():
            print(f"Val/{key}: {value}")
            writer.add_scalar(f"Val/{key}", value, epoch_idx + 1)
        writer.flush()

    def _write_activation_map(self, writer: SummaryWriter, target_layer, inputs, labels, epoch_idx):
        # Create GradCAM object
        cam = GradCAM(model=self, target_layers=[target_layer])

        for i, (input, labels) in enumerate(zip(inputs, labels)):
            # Ensure input_tensor has batch dimension
            if input.ndimension() == 3:  # Shape [C, H, W]
                # Add batch dimension -> [1, C, H, W]
                input = input.unsqueeze(0)
            else:
                raise ValueError(
                    f"Unexpected input_tensor shape: {input.shape}")

            # Create a TargetCategory object
            target = ClassifierOutputTarget(0)

            # Generate Grad-CAM heatmap
            grayscale_cam = cam(input_tensor=input,
                                targets=[target])[0, :]

            # Prepare input image for visualization
            input_image = input[0].cpu().permute(
                1, 2, 0).numpy()  # Convert to [H, W, C]

            # Undo normalization for visualization (if normalization was applied in transforms)
            transform_norm = Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                       std=[1/0.229, 1/0.224, 1/0.225])
            input_image = transform_norm(torch.tensor(
                input_image).permute(2, 0, 1)).permute(1, 2, 0).numpy()
            input_image = (input_image - input_image.min()) / \
                (input_image.max() - input_image.min())  # Normalize to [0, 1]

            # Overlay Grad-CAM heatmap on the input image
            cam_image = show_cam_on_image(
                input_image, grayscale_cam, use_rgb=True)

            # Convert to [C, H, W] for TensorBoard
            cam_image = np.transpose(cam_image, (2, 0, 1))

            # Log to TensorBoard
            writer.add_image(
                f"Activation Map/Label_{labels.item()}_Index_{i}", cam_image, epoch_idx)
        writer.flush()
