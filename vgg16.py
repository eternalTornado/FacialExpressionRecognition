import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2


config = tf.ConfigProto(device_count = {"GPU": 0, "CPU": 56})
sess = tf.Session(config = config)
keras.backend.set_session(sess)

num_classes = 7
#batch_size = 125
batch_size = 16
epochs = 10

class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

with open("./fer2013/fer2013.csv") as f:
	content = f.readlines()
lines = np.array(content)
num_of_instances = lines.size
print("num of instances: ", num_of_instances)
print("instance length: ",len(lines[1].split(",")[1].split(" ")))

train_images, train_labels, test_images, test_labels = [], [], [], []

for i in range(num_of_instances):
	try:
		emotion, img, usage = lines[i].split(",")
		val = img.split(" ")
		pixels = np.array(val, "float32")
		emotion = keras.utils.to_categorical(emotion, num_classes)
		if "Training" in usage:
			train_labels.append(emotion)
			train_images.append(pixels)
		elif "PublicTest" in usage:
			test_images.append(pixels)
			test_labels.append(emotion)
	except:
		print("",end="")

train_images = np.array(train_images, "float32")
train_labels = np.array(train_labels, "float32")
test_images = np.array(test_images, "float32")
test_labels = np.array(test_labels, "float32")

train_images = train_images/255
test_images = test_images/255


train_images = train_images.reshape(train_images.shape[0], 48,48)
train_images = train_images.astype("float32")
test_images = test_images.reshape(test_images.shape[0], 48,48)
test_images = test_images.astype("float32")

train_images_vgg16 = []
test_images_vgg16 = []

def to_rgb(img):
	img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
	return img_rgb

# #pre-process data to 3-channeled color
# for i in range(0, len(train_images)):
# 	rgb = to_rgb(train_images[i])
# 	train_images_vgg16.append(rgb)

# train_images_vgg16 = np.stack([train_images_vgg16], axis=4)
# train_images_vgg16 = np.squeeze(train_images_vgg16, axis=4)
# print(train_images_vgg16.shape)

# for i in range(0, len(test_images)):
# 	rgb = to_rgb(test_images[i])
# 	test_images_vgg16.append(rgb)
# test_images_vgg16 = np.stack([test_images_vgg16], axis=4)
# test_images_vgg16 = np.squeeze(test_images_vgg16, axis=4)
# print(train_images_vgg16.shape)

# train_images_vgg16 = np.array(train_images_vgg16, "float32")
# test_images_vgg16= np.array(test_images_vgg16, "float32")


# train_images_vgg16 = train_images_vgg16.reshape(train_images_vgg16.shape[0], 48,48)
# train_images_vgg16 = train_images_vgg16.astype("float32")
# test_images_vgg16 = test_images_vgg16.reshape(test_images_vgg16.shape[0], 48,48)
# test_images_vgg16 = test_images_vgg16.astype("float32")

#model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(48,48,3))
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, (3,3), activation="relu", padding="same", input_shape=(48,48,1)))
model.add(keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"))
model.add(keras.layers.MaxPool2D((2,2), strides=(2,2)))

model.add(keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"))
model.add(keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"))
model.add(keras.layers.MaxPool2D((2,2), strides=(2,2)))

model.add(keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"))
model.add(keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"))
model.add(keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"))
model.add(keras.layers.MaxPool2D((2,2), strides=(2,2)))

model.add(keras.layers.Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(keras.layers.Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(keras.layers.Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(keras.layers.MaxPool2D((2,2), strides=(2,2)))

model.add(keras.layers.Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(keras.layers.Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(keras.layers.Conv2D(512, (3,3), activation="relu", padding="same"))
model.add(keras.layers.MaxPool2D((2,2), strides=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(4096, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(4096, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation="softmax"))

gen = keras.preprocessing.image.ImageDataGenerator()
train_generator = gen.flow(train_images_vgg16, train_labels, batch_size=batch_size)

model.compile(
	loss="categorical_crossentropy",
	optimizer=keras.optimizers.Adam(),
	metrics=["accuracy"]
	)


model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs = epochs)
#model.fit(train_images_vgg16, train_labels, epochs=epochs, batch_size=batch_size)

train_loss, train_acc = model.evaluate(train_images_vgg16, train_labels);
print("train_loss: ", train_loss)
print("train_acc: ",train_acc)
test_loss, test_acc = model.evaluate(test_images_vgg16, test_labels)
print("test_loss: ", test_loss)
print("test_acc: ", test_acc)

model.save("face_model_vgg16.h5")
del model

