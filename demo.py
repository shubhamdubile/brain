from pickle import FRAME
from queue import Empty

import streamlit as st
import cv2 
import pandas as pd

import numpy as np 
import streamlit as st
import tensorflow as tf

def predict(image):
    classifier_model = "new_vgg19.h5"
    IMAGE_SHAPE = (224, 224,3)
    model = tf.keras.models.load_model(classifier_model)
    test_image = cv2.resize(image, (224,224),interpolation = cv2.COLOR_RGB2BGR)
    
    test_image = test_image.reshape(1,224,224,3)
    class_names = [
          'glioma',
          'no',
          'Menginioun', 
          'Pituratry'
          ]
    predictions = model.predict(test_image)
    
    scores=np.argmax(predictions, axis=1)
    results = class_names[scores]

    
    
    return results

uploaded_file = st.file_uploader("Choose a image file")
if uploaded_file  is not None:
    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        st.text(predict(uploaded_file))
    



    

