# Tiny ImangeNet Dataloader
import os
import numpy as np
import torch
from PIL import Image
import torch.utils.data


def default_loader(path):
    return Image.open(path).convert('RGB')


class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_list, transform=None, loader=default_loader):
        # root: your_path/TinyImageNet/
        # data_list: your_path/TinyImageNet/train.txt etc.
        images = []
        labels = open(data_list).readlines()
        for line in labels:
            items = line.strip('\n').split()
            img_name = items[0]

            # test list contains only image name
            test_flag = True if len(items) == 1 else False
            label = None if test_flag else np.array(int(items[1]))

            if os.path.isfile(os.path.join(root, img_name)):
                images.append((img_name, label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')

        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return (img, label) if label is not None else img

    def __len__(self):
        return len(self.images)
