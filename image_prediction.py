from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np
from os.path import join
import pandas as pd
import os
from PIL import Image as im


### provide a name of the model in h5 format here
model_name = 'bumper_damage_front_rear_021019_224pix_vgg19_00001.h5'

basedir = os.path.abspath(os.path.dirname(__file__))

model_path = os.path.join(basedir, model_name)

class prediction:

    def __init__(self):
        self.my_model = load_model(filepath=model_path)

    def get_prediction(self, image_path):

        img = image.load_img(path=image_path, grayscale=False, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        print(x)


        prediction = self.my_model.predict(x)

        return prediction