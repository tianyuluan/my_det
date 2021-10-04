#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-10-02 21:12:48
LastEditTime: 2021-10-04 09:33:04
motto: Still water run deep
Description: Modify here please
FilePath: /my_det/head/anchor_free_head.py
'''
import torch
import torch.nn as nn
from utils import multi_apply
from loss import GFocalLoss
from loss import GIoULoss

class AnchorFreeHead(nn.Module):
    def __init__(self, 
                              in_channel=256,
                              feat_channel=256,
                              num_classes=80,
                              ):
        super(AnchorFreeHead, self).__init__()
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = GFocalLoss()
        self.loss_iou  = GIoULoss()
        # self.
    
    def _build_head(self, in_channel, feat_channel, out_channel):
        head = nn.Sequential(
                nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_channel, out_channel, kernel_size=1)
        )
        return head

    # def init_weights(self, ):
    #     pass
    
    def forward(self, features):
        return multi_apply(self.single_forward, features, 
                                                    [i for i in range(len(features))])

    def single_forward(self, feature, i):
        center_heatmap_pred = self.heatmap_head(feature).sigmoid()
        wh_pred = self.wh_head(feature)
        offset_pred = self.offset_head(feature)
        return center_heatmap_pred, wh_pred, offset_pred

    def loss(self,
                     center_heatmap_pred,
                     wh_pred,
                     offset_pred,
                     gt_labels,
                     gt_bboxes):
        pass

    def get_targets(self, 
                                     gt_bboxes,
                                     gt_labels,
                                     feature_shape,
                                     img_shape):
        img_h, img_w = img_shape[:2]
        bs, _, feature_h, feature_w = feature_shape

        w_ratio = float(img_w / feature_w)
        h_ratio = float(img_h / feature_h)
        