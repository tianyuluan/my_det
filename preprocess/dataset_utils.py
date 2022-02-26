#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2022-02-21 21:49:19
LastEditTime: 2022-02-25 14:11:38
motto: Still water run deep
Description: Modify here please
FilePath: /my_det/preprocess/dataset_utils.py
'''
from functools import wraps
from torch.utils.data.dataset import Dataset as torchDataset

class Dataset(torchDataset):
    """
    This is an basic dataset class for MyDataset
    """
    def __init__(self, input_dim, mosaic=True):
        super.__init__()
        self.input_dim = input_dim
        self.mosaic = mosaic

    @property
    def _input_dim_(self, ):
        """
        set for transfom data aug"""
        return self.input_dim