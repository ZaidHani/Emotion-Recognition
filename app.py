# importing libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential

import streamlit as st
import keras
import numpy as np


# Define the model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# load the model
model.load_weights('model_weights.h5')

# set webpage
st.title('Emotion Detiction')
testing_image = st.camera_input('Say Cheese!',)

# the classes are ordered alphbaticlly
image_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

try:
    img = tf.keras.utils.load_img(testing_image, target_size=(48,48), color_mode='grayscale')
    #st.image(img)
    img_array = keras.utils.img_to_array(img)
    img_array = keras.backend.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    fixed_predictions = predictions.astype('c')
    #st.markdown(predictions)

    st.markdown(f'This image belongs to the class: <h1>{image_classes[np.argmax(fixed_predictions[0])]}</h1>',
                unsafe_allow_html=True)
except Exception as e:
    if str(e) in "TypeError: path should be path-like or io.BytesIO, not <class 'NoneType'>":
        pass
    else:
        st.markdown(e)
