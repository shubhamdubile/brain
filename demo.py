import streamlit as st
from PIL import Image

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
import cv2

fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Bag Classifier')

st.markdown("Welcome to this simple web application that classifies bags. The bags are classified into six different classes namely: Backpack, Briefcase, Duffle, Handbag and Purse.")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = cv2.imread(file_uploaded,1)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    classifier_model = "new_vgg19.h5"
    IMAGE_SHAPE = (224, 224,3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
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
    results = class_names[scores]

    
    #result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return results









    

if __name__ == "__main__":
    main()
