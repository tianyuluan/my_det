#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-10-04 11:48:54
LastEditTime: 2022-02-26 14:17:51
motto: Still water run deep
Description: Modify here please
FilePath: /my_det/preprocess/dataset.py
'''
import os
import torch
import numpy as np
from .dataset_utils import Dataset
from pycocotools.coco import COCO


class MyDataset(Dataset):
    """
    a COCO format dataset class"""
    def __init__(
        self,
        data_dir=None,
        label_file=None,
        image_size=None,
        data_aug=None,
        ):
        """
        Args:
            data_dir: the root path of images
            label_file: the realpath of label file.(.json)
            data_aug: data augmentation strategy
        """

        super.__init__(image_size)
        self.data_dir = data_dir
        self.label_file = label_file
        self.image_size = image_size
        self.data_aug = data_aug
        """
        COCO(json file path) return a COCO.CLASS
            getImagIds(): return a list  which contain all image id. Such as [0, 1, ....., 69863] --> BDD100k.
            getCatsIds(): return a list which contain the class_number. Such as [0, 1, 2, ...., 10] --> BDD100k.
            loadCats():   return a list[dict], in dict is key 'id'、'name'. Such as [{'id':1, 'name':'pedestrian'}, ...., ] --> BDD100k.
            loadImgs(index): return  a list[dict], length is 1, dict: {'file_name': '0000f77c-6257be58.jpg', 'height': 720, 'width': 1280, 'id':1}.
            getAnnIds(imgIds=[int(index)], iscrowd=False): return a anno_ids:list<int>, which is the anno index about frame[index].
            loadAnns(anno_ids): return a list:list<dict>, which contain the anno info about a single frame.
        """
        self.coco = COCO(os.path.join(self.label_file))

        self.ids = self.coco.getImagIds()
        cats = self.coco.loadCats(self.coco.getCatsIds())
        self.class_name = tuple([cat['name'] for cat in cats])
        self.class_ids = sorted(self.coco.getCatsIds())
        self.annos = self.load_annos()
        
    def load_annos(self,):
        annos = []
        for id_ in self.ids:
            im_anno = self.coco.loadImgs(id_)[0]
            w = im_anno['width']
            h = im_anno['height']
            anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
            annotations = self.coco.loadAnns(anno_ids)
            objs = []
            for obj in annotations:
                x1 = np.max((0, obj['bbox'][0]))
                y1 = np.max((0, obj['bbox'][1]))
                x2 = np.min((w, x1 + np.max((0, obj['bbox'][2]))))
                y2 = np.min((h, y1 + np.max((0, obj['bbox'][3]))))
                if obj['area'] > 0 and x2>=x1 and y2>=y1:
                    obj['clean_bbox'] = [x1, y1, x2, y2]
                    objs.append(obj)
            obj_num = len(objs)
            res = np.zeros((obj_num, 5))
            for i, obj_ in enumerate(objs):
                cls = self.class_ids.index(obj['category_id'])
                res[i, 0:4] = obj['clean_bbox']
                res[i, 4] = cls
            
            # resize: keep ratio
            ratio = min(self.image_size[0]/h, self.image_size[1]/w)
            res[:, :4] *= ratio

            img_info = (h, w)
            resized_info = (int(h*ratio), int(w*ratio))

            file_name = (
                im_anno['file_name']
            )
            annos.append((res, img_info, resized_info,file_name))
        return annos

    def load_resized_img(self, index):
        image_name = self.annos[index][3]
        image_file = os.path.join(self.data_dir, image_name)
        img = cv2.imread(image_file)
        assert img is not None
        
        # resize: keep ratio
        ratio = min(self.image_size[0]/img.shape[0], self.image_size[1]/img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1]*r), int(img.shape[0]*r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        
        return resized_img

    def __len__(self,):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.ids[index]
        target, img_info, resized_info, _ = self.annos[index]
        img = self.load_resized_img(index)
        img_id = np.array([self.ids[index]])

        if self.data_aug is not None:
            img, target = self.data_aug(img, target)
        return img, target, img_info, img_id
    