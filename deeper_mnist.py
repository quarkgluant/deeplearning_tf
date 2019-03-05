#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:33:39 2019

@author: quark
"""

import tensorflow as tf

# load MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# convert to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# print(y_train.shape)
# scale all input values to between 0 and 1
X_train = X_train / 255.
X_test = X_test / 255.

# define our model
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),                          # 28x28 -> 784x1
#        tf.keras.layers.Dense(1024, activation=tf.nn.relu),  # 784x1 -> 1024x1
#        tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),  # 512x1 -> 512x1
        tf.keras.layers.Dense(512, activation=tf.nn.relu),  # 512x1 -> 512x1
#        tf.keras.layers.Dense(512, activation=tf.nn.relu),  # 512x1 -> 512x1
#        tf.keras.layers.Dense(512, activation=tf.nn.relu),  # 512x1 -> 512x1
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # 512x1 -> 10x1
])

# define our optimizer and loss function
#model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=.05), loss='categorical_crossentropy', metrics=['accuracy'])

# train our model!
model.fit(X_train, y_train, epochs=5, callbacks=[tf.keras.callbacks.TensorBoard(log_dir='deeper_mnist')])

# compute the accuracy for the test set
loss, accuracy = model.evaluate(X_test, y_test)
print('{:.4}'.format(accuracy))
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
#print(sess.run(X_train[0])