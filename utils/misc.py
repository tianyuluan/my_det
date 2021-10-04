#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-10-03 14:43:14
LastEditTime: 2021-10-03 14:44:15
motto: Still water run deep
Description: Modify here please
FilePath: /my_det/utils/misc.py
'''
from functools import partial

import numpy as np
import torch
from six.moves import map, zip

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))