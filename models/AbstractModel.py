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
from torchvision import transforms

from dataset import KagglehubDataset, TransformTorchDataset


class AbstractModel(ABC, nn.Module):
    def __init__(self, name: str):
        super(AbstractModel, self).__init__()
        self.name = name
        self._init_layers()
        self.device = torch.device("cuda")
        self.to(self.device)

    @abstractmethod
    def _init_layers(self):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _create_loss_fn(self) -> _Loss:
        pass

    @abstractmethod
    def _create_optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def _create_dataset(self) -> TorchDataset:
        pass

    @abstractmethod
    def _create_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        '''
        train_transform = transforms.Compose([
          ...
          ])
        val_transform = transforms.Compose([
          ...
        ])
        return (train_transform, val_transform)
        '''
        pass

    def _create_loaders(self) -> tuple[DataLoader, DataLoader]:
        dataset = self._create_dataset()

        train_transform, val_transform = self._create_transforms()

        train_size = 0.7
        val_size = 0.3
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size])

        train_dataset = TransformTorchDataset(
            train_dataset, transform=train_transform)
        val_dataset = TransformTorchDataset(
            val_dataset, transform=val_transform)

        print(f'Training set has {len(train_dataset)} instances')
        print(f'Validation set has {len(val_dataset)} instances')

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        return train_loader, val_loader

    @abstractmethod
    def _train_one_epoch(self, train_loader: DataLoader, loss_fn: _Loss,
                         optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float | int]:
        """
        inputs should be a tensor of shape [B, C, H, W]\n
        labels should be a tensor of shape [B]\n

        return train_summary, inputs, labels
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

    def train_and_save_model(self, epochs: int = 1000, patience: int = 30):
        if epochs < 1:
            raise ValueError("epochs must be greater than 0")

        # Get device
        device = self.device
        print(f"Using device: {device}")

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

        # Write model architecture
        random_image = torch.randn(1, 3, 224, 224).to(device)
        writer.add_graph(self, random_image)

        # Convert random_image to random_image_pil
        random_image_pil = transforms.ToPILImage()(random_image.squeeze(0))

        # Flush
        writer.flush()

        # Create folder to save model
        saved_model_folder_path = os.path.join(
            'trained_models', self.name, timestamp)
        if not os.path.exists(saved_model_folder_path):
            os.makedirs(saved_model_folder_path)

        best_vloss = 1_000_000.
        epochs_since_best = 0
        for epoch_idx in range(epochs):
            print(f"Epoch {epoch_idx + 1} of {epochs}")

            # Train
            self.train()
            train_summary, inputs, labels = self._train_one_epoch(
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
            target_layer = self._get_target_layer()
            self._write_activation_map(
                writer, target_layer=target_layer, inputs=inputs, labels=labels, epoch_idx=epoch_idx, tag="Training activation maps")
            self.eval()
            self._write_activation_map(
                writer, target_layer=target_layer, inputs=vinputs, labels=vlabels, epoch_idx=epoch_idx, tag="Validation activation maps")

            # Track best performance, and save the model's state
            vloss = val_summary['loss']
            if vloss > best_vloss:
                epochs_since_best += 1
                # model_path = os.path.join(
                # saved_model_folder_path, f"{epoch_idx + 1}.pth")
                # torch.save(self.state_dict(), model_path)
            else:
                best_vloss = vloss
                epochs_since_best = 0

            print(f"Expected to stop at epoch {
                  epoch_idx + patience - epochs_since_best + 1} if no further improvements")

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

    def _write_activation_map(self, writer: SummaryWriter, target_layer, inputs, labels, epoch_idx, tag):
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
                f"{tag}/Label_{labels.item()}_Index_{i}", cam_image, epoch_idx)
        writer.flush()
