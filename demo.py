from pickle import FRAME
from queue import Empty
# from turtle import shape
import streamlit as st
import cv2 #computer vision
import pandas as pd
#import cvlib as cv
import numpy as np 
import streamlit as st
import tensorflow as tf
#from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.model import l
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
# from statistics import mode

def predict(image):
    classifier_model = "new_vgg19.h5"
    IMAGE_SHAPE = (224, 224,3)
    model = tf.keras.models.load_model(classifier_model)
    test_image = cv2.resize(image, (224,224),
               interpolation = cv2.COLOR_RGB2BGR)
    # test_image = preprocessing.image.img_to_array(test_image)
    # test_image = test_image / 255.0
    # test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image.reshape(1,224,224,3)
    class_names = [
          'glioma',
          'no',
          'Menginioun', 
          'Pituratry'
          ]
    predictions = model.predict(test_image)
    #scores = tf.nn.softmax(predictions[0])
    #scores = scores.numpy()
    scores=np.argmax(predictions, axis=1)
    st.text(scores[0])
    results = class_names[scores[0]]

    
    #result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return results

uploaded_file = st.file_uploader("Choose a image file")
st.text(uploaded_file)
st.image(uploaded_file)
if uploaded_file  is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
    opencv_image = cv2.imdecode(file_bytes, 1)
    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        st.text(predict(opencv_image))
    



    

