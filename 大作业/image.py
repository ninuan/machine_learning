# 人脸矩形框裁剪
import math
import os
import random

import numpy as np
from PIL import Image, ImageStat, ImageEnhance


def _crop(image, rect):
    """Crop the image w.r.t. box identified by rect"""
    x_min, y_min, x_max, y_max = rect
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    side = max(x_center - x_min, y_center - y_min)
    side *= 1.5
    rect = (x_center - side, y_center - side, x_center + side, y_center + side)
    image = image.crop(rect)
    return image, rect


# 图片缩放
def _resize(image, pts):
    """Resize the image and landmarks simultaneously"""
    target_size = (128, 128)
    pts = pts / image.size * target_size
    image = image.resize(target_size, Image.ANTIALIAS)
    return image, pts


# 统一图片平均亮度
def _relight(image):
    """Standardize the light of an image"""
    r, g, b = ImageStat.Stat(image).mean
    brightness = math.sqrt(0.241 * r ** 2 + 0.691 * g ** 2 + 0.068 * b ** 2)
    image = ImageEnhance.Brightness(image).enhance(128 / brightness)
    return image


# 随机翻转图片
def _fliplr(image, pts):
    """Flip the image and landmarks randomly"""
    if random.random() >= 0.5:
        pts[:, 0] = 128 - pts[:, 0]
        pts = pts[_fliplr.perm]
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image, pts


_fliplr.perm = np.load('fliplr_perm.npy')

def preprocess(dataset:'File',name:str):
    """Preprocess input data as described in dataset"""
    print(f"start preprocess {name}")
    image_dir = './WFLW/WFLW_images/'
    target_base = f'./dataset/{name}'
    os.mkdir(target_base)

    pts_set = []
    batch = 0
    for data in dataset:
        if not pts_set:
            print("\rbatch" + str(batch),end="")
            target_dir = target_base + f'/batch_{batch}/'
            os.mkdir(target_dir)
        data = data.split(' ')
        pts = np.array(data[:196], dtype=np.float32).reshape((98,2))
        rect = [int(x) for x in data[196:200]]
        image_path = data[-1][:-1]

        with Image.open(image_dir + image_path) as image:
            img,rect = _crop(image,rect)
        pts -= rect[:2]
        img,pts = _resize(img,pts)
        img,pts = _fliplr(img,pts)
        ima = _relight(img)

        img.save(target_dir + str(len(pts_set)) + '.jpg')
        pts_set.append(np.array(pts))
        if len(pts_set) == 50:
            np.save(target_dir + 'pts.npy',pts_set)
            pts_set = []
            batch += 1
    print()

if __name__ == '__main__':
    annotation_dir = './WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/'
    train_file = 'list_98pt_rect_attr_train.txt'
    test_file = 'list_98pt_rect_attr_test.txt'
    _fliplr.perm = np.load('fliplr_perm.npy')

    os.mkdir('./dataset')
    with open(annotation_dir + train_file,'r') as dataset:
        preprocess(dataset,'train')
    with open(annotation_dir + test_file,'r') as dataset:
        preprocess(dataset,'test')