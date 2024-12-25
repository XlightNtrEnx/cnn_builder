from torch.utils.data import Dataset as TorchDataset
import os
from PIL import Image as PILImage


class Dataset(TorchDataset):
    def __init__(self, label: int, img_dir: str, max_size: int, transform: callable = None):
        super(Dataset, self).__init__()
        self.label = label
        self.transform = transform
        self.img_paths = self._get_img_paths(img_dir, max_size)

    def __getitem__(self, idx):
        img = PILImage.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label

    def __len__(self):
        return len(self.img_paths)

    def _get_img_paths(self, img_dir: str, max_size: int):
        valid_extensions = ('.jpg', '.jpeg', '.png')
        img_paths = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(valid_extensions):
                    img_paths.append(os.path.join(root, f))
                    if len(img_paths) >= max_size:
                        break
            if len(img_paths) >= max_size:
                break
        return img_paths
