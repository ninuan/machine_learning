import os

import numpy as np
from PIL import Image

from keras.applications import ResNet50
# from keras.integration_test.preprocessing_test_utils import BATCH_SIZE
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import plot_model
import matplotlib.pyplot as plt
BATCH_SIZE = 50


def pretrain(model, name):
    """Use a pretrained model to extract features"""
    print('predicting on ' + name)
    base_path = f'./dataset/{name}/'
    for batch_path in os.listdir(base_path):
        batch_path = base_path + batch_path + '/'
        images = np.zeros((50, 128, 128, 3), dtype=np.uint8)
        for i in range(50):
            with Image.open(batch_path + f'{i}.jpg') as image:
                images[i] = np.array(image)
        result = model.predict_on_batch(images)
        np.save(batch_path + 'resnet50.npy', result)


base_model = ResNet50(include_top=False, input_shape=(128, 128, 3))
output = base_model.layers[38].output
model = Model(inputs=base_model.input, outputs=output)
pretrain(model, 'train')
pretrain(model, 'test')

model = Sequential()
model.add(Conv2D(256, (1, 1), input_shape=(32, 32, 256), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(512, (2, 2), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(196))
model.compile('adam', loss='mse', metrics=['accuracy'])
model.summary()
plot_model(model, to_file='./models/model.png', show_shapes=True)


def data_generate(base_path):
    """Data generator for keras fitter"""
    while True:
        for batch_path in os.listdir(base_path):
            batch_path = base_path + batch_path + '/'
            pts = np.load(batch_path + 'pts.npy').reshape((BATCH_SIZE, 196))
            _input = np.load(batch_path + 'resnet50.npy')
            yield _input, pts


train_generator = data_generate('./dataset/train/')
test_generator = data_generate('./dataset/test/')

history = model.fit_generator(train_generator, steps_per_epoch=150, epochs=4, validation_data=test_generator,
                              validation_steps=50)
model.save('./models/model.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./models/accuracy.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./models/loss.png')
plt.show()
