#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-10-02 21:12:48
LastEditTime: 2021-10-06 10:33:57
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
        return multi_apply(self.single_forward, features, [i for i in range(len(features))])

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
        
        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feature_h, feature_w])
        wh_target = gt_bboxes[-1].new_zeros(
            [bs, 2, feature_h, feature_w])
        offset_target = gt_bboxes[-1].new_zeros(
            [bs, 2, feature_h, feature_w])
        wh_offset_target_weight = gt_bboxes.new_zeros(
            [bs, 2, feature_h, feature_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * w_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * h_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim =1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bboxes[j][3] - gt_bboxes[j][1]) * h_ratio
                scale_box_w = (gt_bboxes[j][2] - gt_bboxes[j][0]) * w_ratio         
                radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
                 
                radius = max(0, int(radius))
                ind = gt_labels[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                                              [ctx_int, cty_int], radius)
                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, cty_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, cty_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

    avg_factor = max(1, center_heatmap_target.eq(1).sum())
    target_result = dict(
        center_heatmap_target=center_heatmap_target,
        wh_target=wh_target,
        offset_target=offset_target,
        wh_offset_target_weight=wh_offset_target_weight
    )
    return target_result, avg_factor


    def gaussian_radius(self, det_size, min_overlap):
        pass

    def gen_gaussian_target(self, heatmap, center, radius, k=1):
        pass
