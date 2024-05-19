import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
#model = load_model("C:\Users\iaade\PycharmProjects\Animla classiications\Untitled.ipynb")
st.header("Image Classification Model")
model = load_model("C:/Users/iaade/PycharmProjects/Animla classiications/Image_classify.keras")
#put all the fruit name in an array.
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon', 
 "Aubergine",
 "avocado"]
 

img_height = 180

img_width = 180
# create a place holder
img = st.text_input("Enter Image name","apple.jpg")

#Note copy the image.classify.keras
#img = "C:/Users/iaade/PycharmProjects/Animla classiications/apple.jpg"

image_load = tf.keras.utils.load_img(img,target_size =(img_width,img_height))

img_arr  = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(img)
st.write("Veg/Fruit in image is " +" " + data_cat[np.argmax(score)])
st.write("With Accuracy of " + " " +str(np.max(score)*100))