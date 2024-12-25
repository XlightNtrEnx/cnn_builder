import kagglehub
import os
import shutil

from .Dataset import Dataset


class KagglehubDataset(Dataset):
    def __init__(self, kagglehub_url: str, label: int, max_size: int, transform: callable = None):
        self.dataset_path = self._ensure_downloaded(kagglehub_url)
        super().__init__(img_dir=self.dataset_path,
                         label=label, max_size=max_size, transform=transform)

    def _ensure_downloaded(self, kagglehub_url):
        dataset_path = os.path.join(
            "kagglehub_datasets", kagglehub_url)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            cache_path = kagglehub.dataset_download(kagglehub_url)
            print(f"Downloaded dataset to cache at {cache_path}")
            print(f"Moving dataset to {dataset_path}")
            for item in os.listdir(cache_path):
                source = os.path.join(cache_path, item)
                destination = os.path.join(dataset_path, item)
                shutil.move(source, destination)
        return dataset_path
