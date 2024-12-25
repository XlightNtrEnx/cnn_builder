from torch.utils.data import Dataset as TorchDataset


class TransformTorchDataset(TorchDataset):
    def __init__(self, dataset: TorchDataset, transform: callable = None):
        self._dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self._dataset.__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self._dataset)
