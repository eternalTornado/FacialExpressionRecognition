import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


model = keras.models.load_model(face_model.h5)
predictions = model.predict()
