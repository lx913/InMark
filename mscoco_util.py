import torch
import torchvision
from torch.utils.data import Dataset

class CoCo_Dataset(Dataset):
    def __init__(self, root, annFile, transform):
        self.dataset = torchvision.datasets.CocoCaptions(root, annFile)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, captions = self.dataset[idx]

        if self.transform is not None:
            img = self.transform(img)

        if isinstance(captions, (list, tuple)) and len(captions) > 0:
            caption = captions[0]
        else:
            caption = ""
        return img, caption