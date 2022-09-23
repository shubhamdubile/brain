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


st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
    <nav class="navbar navbar-expand-lg navbar-light bg-light">
        <div class="container-fluid">
            <!-- <a class="navbar-brand" href="#">Navbar</a> -->
            <a class="navbar-brand" href="#">
                <img src="https://www.dailyrounds.org/blog/wp-content/uploads/2015/05/caduceus.jpg" width="30" height="30" alt="">
              </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent"
                aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarSupportedContent">
                <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                    <li class="nav-item">
                        <a class="nav-link active" aria-current="page" href="#">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" href="about">Analysis</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" aria-current="page" href="heart">Heart</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" aria-current="page" href="index">Brain</a>
                    </li>
                </ul>
               
            </div>
        </div>
    </nav>
""", unsafe_allow_html=True)


def predict(image):
    classifier_model = "new_vgg19.h5"
    IMAGE_SHAPE = (224, 224,3)
    model = tf.keras.models.load_model(classifier_model)
    test_image = cv2.resize(image, (224,224)
                            ,interpolation = cv2.COLOR_RGB2BGR
                           )
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

if uploaded_file  is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(uploaded_file)
    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        st.text(predict(opencv_image))
    


