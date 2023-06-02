import os

import numpy as np
import functools

from PIL import Image, ImageDraw
from keras.models import load_model

def _preview(image,pts,r=1,color=(255,0,0)):
    """Draw landmark points on image"""
    draw = ImageDraw.Draw(image)
    for x,y in pts:
        draw.ellipse((x-r,y-r,x+r,y+r),fill=color)

def _reuslt(name,model):

    path = f'./dataset/{name}/batch_0/'
    _input = np.load(path + 'resnet50.npy')
    pts = model.predict(_input)
    for i in range(50):
        with Image.open(path+f'{i}.jpg') as image:
            _preview(image, pts[i].reshape((98, 2)))
            output_dir = f'./visualization/{name}/'
            os.makedirs(output_dir, exist_ok=True)  # 自动创建目录
            output_path = os.path.join(output_dir, f'{i}.jpg')
            image.save(output_path)


train_result = functools.partial(_reuslt,'train')
test_result = functools.partial(_reuslt,'test')

model = load_model('./models/model.h5')
train_result(model)
test_result(model)

