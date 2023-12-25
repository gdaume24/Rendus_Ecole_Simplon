import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
from sklearn import preprocessing
import json

st.set_page_config(page_title="Streamlit eye",
                   page_icon="Shark",
                   )

image = st.file_uploader("Choisissez une image d'oeil à scanner")

def formatting_image_to_model(image):
    image = Image.open(image)
    image = np.asarray(image).astype("float32")
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return image

def predict_image_side(image_formatted):

    model1 = tf.keras.models.load_model(r'C:\VSCODE\iris\modeles\oeil_gauche_droite_model')
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load(r'C:\VSCODE\iris\modeles\labelsencoder\classes.npy')
    prediction = model1.predict(image_formatted)
    prediction = np.argmax(prediction, keepdims = True)
    prediction = np.ravel(prediction)
    prediction = encoder.inverse_transform(prediction)

    return prediction

def predict_person_image(image_formatted, side):

    if side == "left":
        model = tf.keras.models.load_model(r'C:\VSCODE\iris\modeles\modelgauche.hdf5')
    if side == "right":
        model = tf.keras.models.load_model(r'C:\VSCODE\iris\modeles\modelgauche.hdf5')

    encoderside = preprocessing.LabelEncoder()
    encoderside.classes_ = np.load(r'C:\VSCODE\iris\modeles\labelsencoder\coté.npy')
    prediction = model.predict(image_formatted)
    prediction = np.argmax(prediction, keepdims = True)
    prediction = np.ravel(prediction)
    prediction = encoderside.inverse_transform(prediction)

    return prediction

def show_image(image):

    image = Image.open(image)
    return image

if image is not None:
    col1, col2 = st.columns(2)
    image_to_show = show_image(image)
    with col1:
        st.image(image_to_show)

    image_formatted = formatting_image_to_model(image)

    side = predict_image_side(image_formatted)
    number = predict_person_image(image_formatted, side)

    with open('C:\VSCODE\iris\CAS_PRATIQUES\employees_info.json', 'r', encoding = "utf-8") as f:
        data = json.load(f)
    
    information_salarie = data[str(number[0])]
    with col2:
        st.text("")
        st.text("")
        st.text("")
        for key, value in information_salarie.items():
            information = key + " : " + str(value)
            st.text(information)