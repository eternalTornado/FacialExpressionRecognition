import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix

config = tf.ConfigProto(device_count = {"GPU": 0, "CPU": 56})
sess = tf.Session(config = config)
keras.backend.set_session(sess)

num_classes = 7
batch_size = 125
epochs = 35

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


train_images = train_images.reshape(train_images.shape[0], 48,48,1)
train_images = train_images.astype("float32")
test_images = test_images.reshape(test_images.shape[0], 48,48,1)
test_images = test_images.astype("float32")

model = keras.models.load_model("face_model_test.h5")

#Commented lines are for training new model

# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(64, (5,5), activation="relu", input_shape=(48,48,1)))
# model.add(keras.layers.MaxPool2D(pool_size=(5,5), strides=(2,2)))

# model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
# model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
# #model.add(keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
# model.add(keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2,2)))

# model.add(keras.layers.Conv2D(128, (3,3), activation="relu"))
# model.add(keras.layers.Conv2D(128, (3,3), activation="relu"))
# #model.add(keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
# model.add(keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2,2)))

# model.add(keras.layers.Flatten())

# model.add(keras.layers.Dense(1024, activation="relu"))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(1024, activation="relu"))
# model.add(keras.layers.Dropout(0.5))

# model.add(keras.layers.Dense(num_classes, activation="softmax"))

# #gen = keras.preprocessing.image.ImageDataGenerator()
# #train_generator = gen.flow(train_images, train_labels, batch_size=batch_size)

# model.compile(
# 	loss="categorical_crossentropy",
# 	optimizer=keras.optimizers.Adam(),
# 	metrics=["accuracy"]
# 	)

# #model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs = epochs)
# model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)

# train_loss, train_acc = model.evaluate(train_images, train_labels);
# print("train_loss: ", train_loss)
# print("train_acc: ",train_acc)
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("test_loss: ", test_loss)
# print("test_acc: ", test_acc)

# model.save("face_model_test.h5")
# del model



#Confusion matrix
pred_list = []
actual_list = []
predictions = model.predict(train_images)
for i in predictions:
	pred_list.append(np.argmax(i))
for i in train_labels:
	actual_list.append(np.argmax(i))

print(confusion_matrix(actual_list,pred_list))
