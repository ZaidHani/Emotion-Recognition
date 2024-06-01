# importing libraries
import tensorflow as tf
import streamlit as st
import keras
import numpy as np
import cv2

# load the model
with open('final_model.json', 'r') as json_string:
    json_string = json_string.read()
model = tf.keras.models.model_from_json(json_string, custom_objects=None)
model.load_weights('final_weights.h5')

st.title('Emotion Recognition')
testing_image = st.camera_input('Say Cheese!',)

# the classes are ordered alphbaticlly
image_classes = ['anger', 'fear', 'happy', 'neutral', 'sad', 'surprise']

try:
    img = tf.keras.utils.load_img(testing_image, target_size=(48,48), color_mode='grayscale')
    #st.image(img)
    img_array = keras.utils.img_to_array(img)
    img_array = keras.backend.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    for i in range(len(image_classes)):
        st.markdown(f'{image_classes[i]} confidance: {predictions[0][i]}')

    st.markdown(f'This image belongs to the class: <h1>{image_classes[np.argmax(predictions[0])]}</h1>',
                unsafe_allow_html=True)
except Exception as e:
    if str(e) in "TypeError: path should be path-like or io.BytesIO, not <class 'NoneType'>":
        pass
    else:
        st.markdown(e)