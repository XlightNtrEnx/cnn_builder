import torch

from models import BinaryModel


if __name__ == "__main__":
    # Check if GPU is available
    if not torch.cuda.is_available():
        raise Exception("GPU not available")
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name()}")

    model = BinaryModel()
    model.train_and_save_model()
