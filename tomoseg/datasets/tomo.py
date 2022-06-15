# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

from tifffile import imread
from .base import BaseDataset


class TomoDataset(BaseDataset):
    def __init__(self,
                 split,
                 list_path,
                 num_classes=4,
                 num_input_channels=1,
                 preprocessing_fn=None,
                 multi_scale=True,
                 ignore_label=-1,
                 base_size=520,
                 crop_size=(520, 520),
                 scale_factor=11):
        super(TomoDataset, self).__init__(
            num_input_channels, num_classes, ignore_label, base_size, crop_size, scale_factor, preprocessing_fn
        )

        self.split = split
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.img_list = [line.strip().split() for line in open(list_path)]

        self.files = self.read_files()

    def read_files(self):
        files = []
        if self.split == 'unlabeled':
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    'img': image_path[0],
                    'name': name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    'img': image_path,
                    'label': label_path,
                    'name': name,
                })
        return files[:500]

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item['name']

        image = imread(item['img'])
        size = image.shape

        if self.split == 'unlabeled':
            image = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8
            )
            image = self.input_transform(image)
            if image.ndim == 2:
                image = np.expand_dims(image, 0)
            else:
                image = image.transpose((2, 0, 1))
            return image.copy(), np.array(size), name

        label = imread(item['label'])

        if self.split == 'val':
            image, label = self.resize_short_length(
                image,
                label=label,
                short_length=self.base_size,
                fit_stride=8
            )
            image, label = self.generate_validation_sample(image, label)
            return image.copy(), label.copy(), np.array(size), name
        elif self.split == 'test':
            image = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8
            )
            image = self.input_transform(image)

            if image.ndim == 2:
                image = np.expand_dims(image, 0)
            else:
                image = image.transpose((2, 0, 1))
            return image.copy(), label.copy(), np.array(size), name

        image, label = self.resize_short_length(image, label, short_length=self.base_size)
        image, label = self.generate_training_sample(image, label, self.multi_scale)

        return image.copy(), label.copy(), np.array(size), name