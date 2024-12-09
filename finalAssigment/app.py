import streamlit as st
import keras
import numpy as np
import pickle
import pandas as pd


flower_names_df = pd.read_csv('oxford_flower_102_name.csv')
flower_meanings_df = pd.read_csv('language-of-flowers.csv')

def locateMeaning(flowerName) :
    location = np.where(flower_meanings_df == flowerName)
    row, col = location[0], location[1]
    
    if len(row) <= 0:
        st.write("This flower's meaning is currently not documented")
    else:
        for index in range(len(row)):
            name = flower_meanings_df.at[row[index], "Flower"]
            color = flower_meanings_df.at[row[index], "Color"]
            meaning = flower_meanings_df.at[row[index], "Meaning"]
            if type(color) == str:
                st.write(f"If the {name} is {color}, then this flower's meaning is: {meaning}")
            elif type(color) == float:
                st.write(f"In general, {name} have the flower meaning of: {meaning}")
            else:
                st.write("This flower's meaning is currently not documented")

col1, col2, col3, col4, col5, col6 = st.columns(6)
flower1 = "./flower1.png"
flower2 = "./flower2.png"
flower3 = "./flower3.png"
with col1:
    st.image(flower1, width = 50)
with col2:
    st.image(flower2, width = 50)
with col3:
    st.image(flower3, width = 50)
with col4:
    st.image(flower1, width = 50)
with col5:
    st.image(flower2, width = 50)
with col6:
    st.image(flower3, width = 50)

st.title('Flower Classifer!')
st.subheader("Build using the 102 Oxford Flowers Dataset.")
st.divider()

uploaded_file = st.file_uploader("Upload a picture:")

model_pkl_file = "102flowersModel.pkl" 

with open(model_pkl_file, 'rb') as file: 
    model = pickle.load(file)



if uploaded_file is not None:
    st.image(uploaded_file)
    img = keras.utils.load_img(uploaded_file, target_size=(150, 150))
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    classes = np.argmax(predictions, axis = 1)
   #st.write(classes)
    result = classes[0]
    flowerName = flower_names_df.at[result, "Name"]
    flowerName = flowerName.title()
    st.divider()
    st.subheader(f"This is a {flowerName}")
    locateMeaning(flowerName)