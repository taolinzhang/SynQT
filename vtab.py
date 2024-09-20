import torch.utils.data as data

from PIL import Image
import os
import os.path
from torchvision import transforms
import torch
import json
import h5py


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None, loader=default_loader, embed_path=None):
        self.root = root
        self.imlist = json.load(open(flist, 'r'))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = list(self.imlist[index].items())[0]
        img = self.loader(os.path.join(self.root, "images",  impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imlist)


def get_data(name, batch_size=64):
    root = '../vtab_data/vtab-1k-png/' + name
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_kwargs = {"root": root, "transform": transform}
    loader_kwargs = {"num_workers": 4, "pin_memory": False, "drop_last": False}
    train_kwargs = {"shuffle": True, "batch_size": batch_size, **loader_kwargs}
    val_kwargs = {"shuffle": False, "batch_size": 512, **loader_kwargs}

    train_loader = torch.utils.data.DataLoader(
        ImageFilelist(flist=root + "/train800val200.json", **dataset_kwargs), **train_kwargs)

    val_loader = torch.utils.data.DataLoader(
        ImageFilelist(flist=root + "/test.json", **dataset_kwargs), **val_kwargs)
    return train_loader, val_loader