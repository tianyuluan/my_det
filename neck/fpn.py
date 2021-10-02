#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-10-02 17:14:04
LastEditTime: 2021-10-02 21:06:48
motto: Still water run deep
Description: Modify here please
FilePath: /my_det/neck/fpn.py
'''
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, ):
        super(FPN, self).__init__()
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y
        
    def forward(self, x):
        p5 = self.toplayer(x[-1])
        p4 = self._upsample_add(p5, self.latlayer1(x[-2]))
        p3 = self._upsample_add(p4, self.latlayer2(x[-3]))
        p2 = self._upsample_add(p3, self.latlayer3(x[-4]))

        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return [p2, p3, p4, p5]