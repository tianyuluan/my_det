#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-10-02 21:12:48
LastEditTime: 2021-11-06 14:22:09
motto: Still water run deep
Description: Modify here please
FilePath: /my_det/head/anchor_free_head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import multi_apply
from loss import GFocalLoss
from loss import GIoULoss

class AnchorFreeHead(nn.Module):
    def __init__(self, 
        in_channel=256,
        feat_channel=256,
        num_classes=80,):
        super(AnchorFreeHead, self).__init__()
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.image_metas = {
            'ori_shape': (720, 1280, 3),
            'img_shape': (720, 1280, 3),
            'pad_shape': (720, 1280, 3),
            'scale_factor': array([1., 1., 1., 1.], dtype=float32),
            'batch_input_shape': (720, 1280)}
        self.loss_center_heatmap = GFocalLoss()
        self.loss_iou  = GIoULoss()
        self.topk = 100
    
    def _build_head(self, in_channel, feat_channel, out_channel):
        head = nn.Sequential(
                nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_channel, out_channel, kernel_size=1))
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
        wh_offset_target_weight=wh_offset_target_weight)
    return target_result, avg_factor

    def get_bboxes(self,
        center_heatmap_pred,
        wh_pred,
        offset_pred,
        rescale=True,
        with_nms=False):
        assert len(center_heatmap_pred) == len(wh_pred) == len(
            offset_pred)
        scale_factors = [image_metas['scale_factors']]
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            self.image_metas['batch_input_shape'],
            k=self.topk,
            kernel = None)
        
        

        det
        pass

    def decode_heatmap(self,
        center_heatmap_pred.
        wh_pred,
        offset_pred,
        img_shape,
        k=100,
        kernel=3):
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape
        center_heatmap_pred = self.get_local_maximum(center_heatmap_pred, kernel=kernel)
        *batch_dets, topk_ys, topk_xs = self.get_topk_from_heatmap(center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = self.transpose_and_gather_feat(wh_pred, batch_index)
        offset = self.transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0]/2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1]/2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0]/2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1]/2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]), dim =1 )
        return batch_bboxes, batch_topk_labels        


    def gaussian_radius(self, det_size, min_overlap):
        height, width = det_size
        a1 = 1
        b1 = height * width * (1-min_overlap)/(1+min_overlap)
        sq1 = sqrt(b1**2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = sqrt(b2**2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = sqrt(b3**2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)
        return min(r1, r2, r3)

    def gen_gaussian_target(self, heatmap, center, radius, k=1):
        diameter = radius * 2  + 1
        gaussian_kernel = gaussian_2D(
            radius, sigma = diameter /6, dtype=heatmap.dtype, device=heatmap.device)
        x, y = center
        height, width = heatmap.shape[:2]
        
        pass

    def get_local_maximum(self, heat, kernel=3):
        pad = (kernel -1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax==heat).float()
        return heat * keep

    def get_topk_from_heatmap(self, heat_map, k=100):
        """
        heat_map: [batch, num_class, h, w]
        k
        """
        batch, _, h, w = heat_map.size()
        topk_scores, topk_inds, = torch.topk(heat_map.view(batch, -1), k)
        topk_clses = topk_inds // (h * w)
        topk_inds = topk_inds % (h *w)
        topk_ys = topk_inds // w
        topk_xs = (topk_inds % w).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def gaussian_2D(self, radius, sigma=1, dtype=torch.float32, device='cpu'):
        x = torch.arange(
            -radius, radius+1, dtype=dtype, device=device).view(1, -1)
        y= torch.arange(
            -radius, radius+1, dtype=dtype, device=device).view(1. -1)
        h = (-(x*x+y*y)/(2*sigma*sigma)).exp()
        h[h<torch.finfo(h.dtype).eps*h.max()] = 0
        return h

    def gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).repeat(1,1,dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self.gather_feat(feat, ind)
        return feat
