#coding=utf-8

from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import numpy as np
import requests

import tensorflow as tf
tf.python.control_flow_ops = tf

model = ResNet50(weights='imagenet')

img_path = 'test.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
preds = decode_predictions(preds)

print u'图像分类', u'概率'
for items in preds[0]:
    print items[1], items[2]
    # print translate(items[1]), items[2]

def translate(name):
    url = 'http://fanyi.youdao.com/openapi.do?keyfrom=keras55&key=821526358&type=data&doctype=json&version=1.1&q=' + name
    r = requests.get(url)
    return r.json()['translation']
