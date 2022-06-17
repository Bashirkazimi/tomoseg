import cv2
import numpy as np
import random

import torch
from torch.utils import data
import albumentations as A
import torch.nn.functional as F


class BaseDataset(data.Dataset):
    def __init__(self,
                 num_input_channels=1,
                 num_classes=4,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 scale_factor=16,
                 preprocessing_fn=None):

        self.num_input_channels = num_input_channels
        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.preprocessing_fn = preprocessing_fn

        self.scale_factor = scale_factor

        self.train_aug = A.Compose(
            [
                A.VerticalFlip(p=0.3),
                A.HorizontalFlip(p=0.3),
                A.Transpose(p=0.3),
                A.RandomRotate90(p=0.3),
                A.OneOf(
                    [
                        A.ElasticTransform(p=0.3, alpha=120, sigma=120*0.05, alpha_affine=120 * 0.03),
                        A.GridDistortion(p=0.3),
                        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.3)
                    ],
                    p=0.3
                ),
            ]
        )

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image):
        if self.preprocessing_fn is None:
            return image
        image = np.clip(image, np.percentile(image, 0.5), np.percentile(image, 99.9))
        if self.preprocessing_fn == 'tanh':
            image = (image - np.min(image)) * (1.0 / (np.max(image) - np.min(image)))
            image *= 2
            image -= 1
        elif self.preprocessing_fn == 'sigmoid':
            image = (image - np.min(image)) * (1.0 / (np.max(image) - np.min(image)))
        elif self.preprocessing_fn == 'zero_mean':
            image = (image - np.mean(image)) / np.std(image)
        else:
            raise ValueError(f'preprocessing_fn {self.preprocessing_fn} not recongnized!')
        return image

    def label_transform(self, label):
        return np.array(label).astype('int32')

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=padvalue)
        return pad_image

    def rand_crop(self, image, label=None):
        h, w = image.shape[:2]
        image = self.pad_image(image, h, w, self.crop_size, tuple(float(np.min(image)) for _ in range(self.num_input_channels)))
        if label is not None:
            label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))
        new_h, new_w = image.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        if label is not None:
            label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
            return image, label
        return image

    def center_crop(self, image, label=None):
        h, w = image.shape[:2]
        image = self.pad_image(image, h, w, self.crop_size, tuple(float(np.min(image)) for _ in range(self.num_input_channels)))
        if label is not None:
            label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))

        new_h, new_w = image.shape[:2]
        center_h, center_w = new_h // 2, new_w // 2

        image = image[
            center_h - (self.crop_size[0] // 2):center_h + (self.crop_size[0] // 2),
            center_w - (self.crop_size[1] // 2):center_w + (self.crop_size[1] // 2)
        ]

        if label is not None:
            label = label[
                    center_h - (self.crop_size[0] // 2):center_h + (self.crop_size[0] // 2),
                    center_w - (self.crop_size[1] // 2):center_w + (self.crop_size[1] // 2)
                ]
            return image, label

        return image

    def multi_scale_aug(self, image, label=None, rand_scale=1):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image

        return image, label

    def resize_short_length(self, image, label=None, short_length=None, fit_stride=None, return_padding=False):
        h, w = image.shape[:2]
        if h < w:
            new_h = short_length
            new_w = np.int(w * short_length / h + 0.5)
        else:
            new_w = short_length
            new_h = np.int(h * short_length / w + 0.5)
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = 0, 0
        if fit_stride is not None:
            pad_w = 0 if (new_w % fit_stride == 0) else fit_stride - (new_w % fit_stride)
            pad_h = 0 if (new_h % fit_stride == 0) else fit_stride - (new_h % fit_stride)
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=tuple(float(np.min(image)) for _ in range(self.num_input_channels))
            )

        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if pad_h > 0 or pad_w > 0:
                label = cv2.copyMakeBorder( label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
            if return_padding:
                return image, label, (pad_h, pad_w)
            else:
                return image, label
        else:
            if return_padding:
                return image, (pad_h, pad_w)
            else:
                return image

    def generate_training_sample(self, image, label, multi_scale=True):
        image = self.input_transform(image)
        label = self.label_transform(label)
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)

        augmented = self.train_aug(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image, label = self.rand_crop(image, label)

        if image.ndim == 2:
            image = np.expand_dims(image, 0)
        else:
            image = image.transpose((2, 0, 1))

        return image, label

    def generate_validation_sample(self, image, label):
        image = self.input_transform(image)
        label = self.label_transform(label)

        image, label = self.center_crop(image, label)

        if image.ndim == 2:
            image = np.expand_dims(image, 0)
        else:
            image = image.transpose((2, 0, 1))

        return image, label

    def multi_scale_inference(self, model, image, scales=[1]):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 2.0 / 3.0)
        stride_w = np.int(self.crop_size[1] * 2.0 / 3.0)
        final_pred = torch.zeros([1, self.num_classes, ori_height, ori_width]).cuda()
        padvalue = tuple(float(np.min(image)) for _ in range(self.num_input_channels))
        for scale in scales:
            new_img = self.multi_scale_aug(image=image, rand_scale=scale)
            height, width = new_img.shape[:2]

            if max(height, width) <= np.min(self.crop_size):
                new_img = self.pad_image(new_img, height, width, self.crop_size, padvalue)
                if new_img.ndim == 2:
                    new_img = np.expand_dims(new_img, 0)
                else:
                    new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = model(new_img)
                preds = preds[:, :, 0:height, 0:width]
            else:
                if height < self.crop_size[0] or width < self.crop_size[1]:
                    new_img = self.pad_image(new_img, height, width, self.crop_size, padvalue)
                new_h, new_w = new_img.shape[:2]
                rows = np.int(np.ceil(1.0 * (new_h - self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes, new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        crop_img = new_img[h0:h1, w0:w1]
                        if h1 == new_h or w1 == new_w:
                            crop_img = self.pad_image(crop_img, h1-h0, w1-w0, self.crop_size, padvalue)
                        if crop_img.ndim == 2:
                            crop_img = np.expand_dims(crop_img, 0)
                        else:
                            crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = model(crop_img)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1-h0, 0:w1-w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]

            preds = F.interpolate(preds, (ori_height, ori_width), mode='bilinear', align_corners=True)
            final_pred += preds
        return final_pred