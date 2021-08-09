import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import scipy as sp

image = load_img('1.jpg', target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

origin_image = plt.imread('1.jpg')
height_roomout = origin_image.shape[0]/7
width_roomout = origin_image.shape[1]/7

model = ResNet50(weights="imagenet")
print(model.summary())
new_model = Model(model.inputs, [model.layers[-3].output, model.layers[-1].output])
feature, preds = new_model.predict(image)
index = np.argmax(preds[0])

# decode the ImageNet predictions to obtain the human-readable label
decoded = imagenet_utils.decode_predictions(preds)
(imagenetID, label, prob) = decoded[0][0]
label = "{}: {:.2f}%".format(label, prob * 100)
feature = sp.ndimage.zoom(feature[0], (height_roomout, width_roomout, 1), order=2) # shape: height, width, 7,2048

gap_weight = new_model.layers[-1].get_weights()[0]  # (2048, 1000)
cam_weight = gap_weight[:, index]
cam_output = np.dot(feature, cam_weight)

plt.xlabel(label)
plt.imshow(cam_output, cmap='jet', alpha=0.5)
plt.imshow(origin_image, cmap='jet', alpha=0.5)
plt.show()
