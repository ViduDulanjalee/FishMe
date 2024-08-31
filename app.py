import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

st.header = ('Identify Your Fish Using The FishMe Application')
fish_names = ['Black Sea Sprat',
 'Gilt Head Bream',
 'Horse Mackerel',
 'Red Mullet',
 'Red Sea Bream',
 'Sea Bass',
 'Shrimp',
 'Striped Red Mullet',
 'Trout']

model= load_model('FishMe_Image_Recognition.keras')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path,target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'This is an Image of '+fish_names[np.argmax(result)]+ ' with a accuracy score of '+str(np.max(result)*100)
    return outcome

uploaded_file = st.file_uploader('Upload and Image')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name),'wb') as f:
        f.write(uploaded_file.getbuffer())
        
    st.image(uploaded_file,width = 200)
    
st.markdown(classify_images(uploaded_file))
    
