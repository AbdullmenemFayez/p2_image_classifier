import json
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds



def load_model(path):
    model_path = './' + path
    load__model = tf.keras.models.load_model(path
                                                  ,custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
    

def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image    



def predict(image_path, model, top_k=5):

    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)
    preds = model.predict(processed_test_image)
    probs = - np.partition(-preds[0], top_k)[:top_k]
    classes = np.argpartition(-preds[0], top_k)[:top_k]

    return probs, classes

