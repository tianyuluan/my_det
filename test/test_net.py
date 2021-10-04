#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-10-02 17:31:38
LastEditTime: 2021-10-03 20:32:52
motto: Still water run deep
Description: Modify here please
FilePath: /my_det/test/test_net.py
'''
from backbone import resnet50
from neck import FPN
from head import AnchorFreeHead
import torch

if __name__ == "__main__":
    a = torch.randn((1,3,640,640))
    features = resnet50(a, pretrain=True)
    fpn = FPN()
    print(fpn)
    head = AnchorFreeHead()
    print(head)
    features = fpn(features)
    a = AnchorFreeHead().forward(features)
    from IPython import embed
    embed()