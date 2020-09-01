# Import Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, 
                                     Input, Reshape, Flatten)

# Import helper dependencies
import numpy as np
import matplotlib.pyplot as plt

# Get dataset
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255, X_test/255

# Visualize data
plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Show images
X_train, X_test = X_train/255., X_test/255.
class_names = [0,1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()

# Reshape images
X_train_ = np.reshape(X_train, (-1,28,28,1))
X_test_ = np.reshape(X_test, (-1,28,28,1))
print(X_test_.shape, type(X_test_)==type(X_test))

# building the model
model = keras.Sequential(name='LeNet_1')
model.add(Input(shape=(28,28,1)))
model.add(Conv2D(32, 4, activation=keras.activations.relu, name='Layer1_Conv'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', 
                    name='Layer1_Mpool'))
model.add(Conv2D(64, 4, activation=keras.activations.relu, name='Layer2_Conv'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', 
                    name='Layer2_Mpool'))
model.add(Flatten(name='Layer3_fl'))
model.add(Dense(1024, name='Layer4_fc', activation=keras.activations.relu))
model.add(Dense(10, name='Layer5_fc'))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



model.summary()

history = model.fit(X_train_, y_train, epochs=10)

# Test the model
test_loss, test_acc = model.evaluate(X_test_, y_test, verbose=2)
print("Test accuracy:{:.2f}%".format(test_acc * 100))

# Plot the accuracy and the loss
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label = 'loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
