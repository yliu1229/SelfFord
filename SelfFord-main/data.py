import os
import random

import numpy as np
import cv2
import torchvision

from typing import Optional, Callable
from PIL import Image
from pathlib import Path
from torchvision.datasets import VisionDataset


def pic_compress(pic_path, QF=90):
    """
        Read an image from path and then JPEG compress it with QF
    """
    with open(pic_path, 'rb') as f:
        pic_byte = f.read()

    img_np = np.frombuffer(pic_byte, np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_ANYCOLOR)

    # compress with QF
    pic_byte = cv2.imencode('.jpg', img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), QF])[1]

    # transform to PIL.Image format
    io_buf = cv2.imdecode(pic_byte, -1)
    img_QF = Image.fromarray(cv2.cvtColor(io_buf, cv2.COLOR_BGR2RGB)).convert('RGB')

    return img_QF


class ImageDataset(VisionDataset):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            crop_size = 64,
            patch_num = 2,
            QFs=None
    ):
        super(ImageDataset, self).__init__(root, transforms, transform, target_transform)

        if QFs is None:
            QFs = [90, 70, 60, 50]
        self.transform_crop = torchvision.transforms.RandomCrop(crop_size)
        self.transform_tensor = torchvision.transforms.ToTensor()
        self.patch_num = patch_num

        self.QFs = QFs

        image_dir = os.path.join(root, 'world')
        if not os.path.isdir(root) or not os.path.isdir(image_dir) or not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')

        imgs = os.listdir(image_dir)
        self.images = [os.path.join(image_dir, x) for x in imgs]

        assert all([Path(f).is_file() for f in self.images])

    def __getitem__(self, index: int):
        try:
            img = Image.open(self.images[index]).convert('RGB')
        except:
            print('Image open = ', self.images[index])
            img = Image.open(self.images[index+1]).convert('RGB')
        # get a compressed copy
        img_QF = pic_compress(self.images[index], random.choice(self.QFs))

        patches = []
        for i in range(self.patch_num):
            patch = self.transform_crop(img)
            patch = self.transform_tensor(patch)
            patches.append(patch)

        patch = self.transform_crop(img_QF)
        patch = self.transform_tensor(patch)
        patches.append(patch)

        return patches

    def __len__(self) -> int:
        return len(self.images)
