#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import mnist
import keras

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape) # (60000,)

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dense, Dropout

model = Sequential([
  Dense(64, activation='relu'),
  Dense(64, activation='sigmoid'),
  Dense(10, activation='softmax'),
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=5,
  batch_size=32,
  validation_data=(test_images, to_categorical(test_labels)) 
)

model.evaluate(
  test_images,
  to_categorical(test_labels)
)

model.save_weights('model.h5')

#to load weights later
#model.load_weights('model.h5')

predictions = model.predict(test_images[:25])

print(np.argmax(predictions, axis=1))
print(test_labels[:25])


# In[ ]:




