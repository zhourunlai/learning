from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input

import tensorflow as tf
tf.python.control_flow_ops = tf

model = VGG16(weights='imagenet', include_top=False)

img_path = 'test.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

print features
