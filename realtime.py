import os
import cv2
from PIL import Image
import numpy as np
import pickle

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.regularizers import l2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
sad_cascade = cv2.CascadeClassifier("haarcascade_sad.xml")


model = keras.models.load_model("face_model_test.h5")

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
#emotions = ["negative", "positive", "neutral"]
def detect(gray_img, frame):
	faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
	for(x_face, y_face, width_face, height_face) in faces:
		cv2.rectangle(frame, (x_face, y_face), (x_face+width_face, y_face+height_face), (255,0,0), 2)
		roi_gray = gray_img[y_face:y_face+height_face, x_face:x_face+width_face]
		#roi_frame = frame[y_face:y_face+height_face, x_face:x_face+width_face]
		roi_gray = cv2.resize(roi_gray, (48,48))
		np_face = keras.preprocessing.image.img_to_array(roi_gray)
		np_face = np.expand_dims(np_face, axis = 0)
		np_face = np_face/255
		prediction = model.predict(np_face)
		conf = np.argmax(prediction[0])

		#print(conf)
		if conf > -1:
			#print(labels[conf])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = emotions[conf]
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame, name, (x_face,y_face), font, 1, color, stroke, cv2.LINE_AA)

	#frame = cv2.flip(frame,1)
	return frame

video_capture = cv2.VideoCapture(0)
while True:
	rec, frame = video_capture.read()
	gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	canvas = detect(gray_img, frame)
	cv2.imshow("Video", canvas)
	if cv2.waitKey(1) & 0xFF==ord("q"):
		break
video_capture.release()
cv2.destroyAllWindows()