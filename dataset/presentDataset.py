"""
@author: Qingrui Zhang
@contact: buptzqr@gmail.com
"""

import copy
import cv2
import numpy as np
import random

import torch
from torch.utils.data import Dataset
from lib.utils.transforms import get_affine_transform
from config import cfg
import torchvision.transforms as transforms
from dataset.attribute import load_dataset


class PresentDataset(Dataset):
    # 一般文件的格式是：图片路径 bbox，以空格分隔
    def __init__(self, DATASET, infoPath, transform=None):
        infoFile = open(infoPath, 'r')
        imgs = []
        bboxes = []
        ids = []
        for line in infoFile:
            line = line.rstrip()
            infos = line.split(':')
            imgs.append(infos[0])
            img_id = infos[0].split('/')[-2] + '/' + infos[0].split('/')[-1]
            ids.append(img_id)
            bboxes_pre = infos[1]
            bboxes_pre = bboxes_pre.split('-')
            for box in bboxes_pre:
                box = box.strip('[]')
                if box == "":
                    continue
                box = box.split(',')
                box_list = []
                for box_elem in box:
                    box_list.append(float(box_elem))
            bboxes.append(box_list)
        self.imgs = imgs
        self.transform = transform
        self.bboxes = bboxes
        self.ids = ids
        self.test_x_ext = DATASET.TEST.X_EXTENTION
        self.test_y_ext = DATASET.TEST.Y_EXTENTION
        self.w_h_ratio = DATASET.WIDTH_HEIGHT_RATIO
        self.pixel_std = DATASET.PIXEL_STD
        self.input_shape = DATASET.INPUT_SHAPE
        self.keypoint_num = DATASET.KEYPOINT.NUM

    def _bbox_to_center_and_scale(self, bbox):
        x, y, w, h = bbox

        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w / 2.0
        center[1] = y + h / 2.0

        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
                         dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        rotation = 0
        score = 1
        img_path = self.imgs[index]
        data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if data_numpy is None:
            raise ValueError('fail to read {}'.format(img_path))
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        bbox = self.bboxes[index]
        center, scale = self._bbox_to_center_and_scale(bbox)

        scale[0] *= (1 + self.test_x_ext)
        scale[1] *= (1 + self.test_y_ext)
        # fit the ratio
        if scale[0] > self.w_h_ratio * scale[1]:
            scale[1] = scale[0] * 1.0 / self.w_h_ratio
        else:
            scale[0] = scale[1] * 1.0 * self.w_h_ratio

        trans = get_affine_transform(center, scale, rotation, self.input_shape)

        img = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.input_shape[1]), int(self.input_shape[0])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            img = self.transform(img)
        img_id = self.ids[index]
        return img, score, center, scale, img_id

    def __len__(self):
        return len(self.imgs)

    def visualize(self, img, joints, score=None):
        pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                 [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                 [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        color = np.random.randint(0, 256, (self.keypoint_num, 3)).tolist()
        joints_array = np.ones((self.keypoint_num, 2), dtype=np.float32)
        for i in range(self.keypoint_num):
            joints_array[i, 0] = joints[i * 3]
            joints_array[i, 1] = joints[i * 3 + 1]
            # joints_array[i, 2] = joints[i * 3 + 2]

        for i in range(self.keypoint_num):
            if joints_array[i, 0] > 0 and joints_array[i, 1] > 0:
                cv2.circle(img, tuple(
                    joints_array[i, :2]), 2, tuple(color[i]), 2)

        def draw_line(img, p1, p2):
            c = (0, 0, 255)
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(img, tuple(p1), tuple(p2), c, 2)

        for pair in pairs:
            draw_line(img, joints_array[pair[0] - 1],
                      joints_array[pair[1] - 1])

        return img


if __name__ == '__main__':
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    attr = load_dataset(cfg.DATASET.NAME)
    dataset = PresentDataset(attr, cfg.INFO_PATH, transform)
