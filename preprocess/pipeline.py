#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-10-04 11:49:06
LastEditTime: 2022-02-26 17:45:30
motto: Still water run deep
Description: Modify here please
FilePath: /my_det/preprocess/pipeline.py
'''
import numpy as np



def xyxy2cxcywh(boxes):
    pass

class TrainTransform:
    def __init__(self, max_labels=None, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            return image, targets
        
        image_ = image.copy()
        targets_ = targets.copy()
        h_, w_, _ = image_.shape
        boxes_ = targets[:, :4]
        labels_ = targets[:. 4]


        
