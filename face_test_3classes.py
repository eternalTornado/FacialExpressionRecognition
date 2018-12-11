import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def emotion_detect(emotions):
	objects = ("negative", "positive", "neutral")
	y_pos = np.arange(len(objects))

	plt.bar(y_pos, emotions, align="center", alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel("percentage")
	plt.title("emotion")

	plt.show()

model = keras.models.load_model("face_model_3classes.h5")

img = keras.preprocessing.image.load_img("angry-man.jpg", color_mode = "grayscale", target_size=(48,48))
np_face = keras.preprocessing.image.img_to_array(img)
np_face = np.expand_dims(np_face, axis = 0)

np_face = np_face/255

prediction = model.predict(np_face)
emotion_detect(prediction[0])
print(prediction)
print(np.argmax(prediction[0]))
np_face = np.array(np_face, "float32")
np_face = np_face.reshape([48,48])
plt.gray()
plt.imshow(np_face)
plt.show()