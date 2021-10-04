#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-10-03 14:09:08
LastEditTime: 2021-10-03 20:12:59
motto: Still water run deep
Description: Modify here please
FilePath: /my_det/loss/__init__.py
'''
from .GFocalLoss import GFocalLoss
from .GIoULoss import  GIoULoss
__all__ = ['GFocalLoss', 'GIoULoss']