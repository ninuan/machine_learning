# 人脸矩形框裁剪
import math
from PIL import Image,ImageStat


def _crop(image,rect):
    """Crop the image w.r.t. box identified by rect"""
    x_min,y_min,x_max,y_max = rect
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    side = max(x_center - x_min, y_center - y_min)
    side *= 1.5
    rect = (x_center - side, y_center - side, x_center + side, y_center + side)
    image = image.crop(rect)
    return image,rect

# 图片缩放
def _resize(image,pts):
    """Resize the image and landmarks simultaneously"""
    target_size = (128,128)
    pts = pts / image.size * target_size
    image = image.resize(target_size, Image.ANTIALIAS)
    return image, pts

# 统一图片平均亮度
def _relight(image):
    """Standardize the light of an image"""
    r,g,b = ImageStat.Stat(image).mean
    brightness = math.sqrt(0.241 * r ** 2 + 0/691 * g ** 2 + 0.068 * b ** 2)