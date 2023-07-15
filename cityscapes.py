import json
import os
import random
from collections import namedtuple
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch.nn.functional as F


class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 25, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 24, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 23, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 22, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 21, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 20, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 19, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 18, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 13, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 13, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, 13, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, root, cell_list, interested_classes, split='train', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []
        self.cell_list = cell_list
        self.interested_classes = interested_classes

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    @classmethod
    def encode_cell(cls, target, cell_list, interested_classes):
        label_len = len(cell_list) * len(interested_classes)
        label_list = np.zeros(label_len, dtype=np.uint8)

        for num, region in enumerate(cell_list):
            cell = target[region[0][1]:region[1][1], region[0][0]:region[1][0]]
            value, count = np.unique(cell, return_counts=True)
            # print(value, count)

            last_class_idx = len(interested_classes) - 1
            for j, subclasses in enumerate(interested_classes):
                if isinstance(subclasses, int):
                    subclasses = (subclasses,)
                if j == last_class_idx:
                    if any(label_list[num * len(interested_classes):num * len(interested_classes) + last_class_idx]):
                        if np.size(value) == 2 and value[0] == 0 and value[1] == 255 and num >= 216:
                            label_list[num * len(interested_classes) + j] = 1
                        else:
                            label_list[num * len(interested_classes) + j] = 0
                    else:
                        if np.size(value) == 1 and value[0] == 255 and num <= 216:
                            label_list[num * len(interested_classes) + j] = 0
                        else:

                            label_list[num * len(interested_classes) + j] = 1
                else:
                    if any(subclass in value for subclass in subclasses):
                        label_list[num * len(interested_classes) + j] = 1
        return label_list

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        filename, _ = os.path.splitext(os.path.basename(self.images[index]))
        target = Image.open(self.targets[index])
        cell_list = self.cell_list
        interested_classes = self.interested_classes
        # flip image and target image horizontally with 0.5 probability
        if self.split == 'train':
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                new_width = 2198
                new_height = 1099
                image = image.resize((new_width, new_height), Image.ANTIALIAS)
                target = target.resize((new_width, new_height), Image.NEAREST)
                crop_width, crop_height = 2048, 1024
                left = random.randint(0, new_width - crop_width)
                upper = random.randint(0, new_height - crop_height)
                right = left + crop_width
                lower = upper + crop_height
                # Crop the image
                image = image.crop((left, upper, right, lower))
                target = target.crop((left, upper, right, lower))
        if self.transform:
            image = self.transform(image)
        target = self.encode_target(target)
        cell_label = self.encode_cell(target, cell_list, interested_classes)
        image = np.array(image)
        return image, cell_label

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)

