import geemap.foliumap as geemap
import ee
import datetime
import os
import streamlit as st
import time 

html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Soil Prediction 📡 </h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

Map = geemap.Map(zoom=1,Draw_export=True)

def maskS2clouds(image):
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloudBitMask = 1 << 10
  cirrusBitMask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))

  return image.updateMask(mask).divide(10000)


l8 = ee.ImageCollection('COPERNICUS/S2').filterDate('2023-01-01', '2023-03-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(maskS2clouds)
# l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1').filterDate('2017-01-01', '2017-12-31')

Map.setCenter(78.9629, 20.5937, 8)
Map.addLayer(l8.median(), {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3}, "Copernicus")
Map.to_streamlit()

from prediction import *

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0      

st.subheader("Determine the most suitable soil type using satellite images 🥕")
uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        prediction = soil_predictor(os.path.join('uploaded',uploaded_file.name))
        # st.text("Prediction: " + prediction)
        st.success(f"{prediction} is predicted by the model for the given image.")
        # print(prediction)
        os.remove('uploaded/'+uploaded_file.name)

        if(prediction == "Alluvial Soil"):
            st.success("Rice, Sugarcane, Tobacco, Maize, Soybean, Jute are the crops recommended for this type of soil")
        if(prediction == "Black Soil"):
            st.success("Groundnut and Cotton are the crops recommended for this type of soil")
        if(prediction == "Desert Soil"):
            st.success("Cactus, Agave or Mesquite are the crops recommended for this type of soil")
        if(prediction == "Red Soil"):
            st.success("Millets, Tobacco, Oil seeds, Potatoes are the crops recommended for this type of soil")

# out_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
# filename = os.path.join(out_dir, 'data.png')

# while feature is None:
#     feature = Map.draw_last_feature
#     time.sleep(5)

# roi = feature.geometry()

# image = l8.clip(roi)

# image = image.unmask()
# geemap.ee_export_image(
#     image, filename=filename, scale=90, region=roi, file_per_band=False
# )

# geemap.ee_export_image(
#     image, filename=filename, scale=90, region=roi, file_per_band=True
# )

import streamlit as st 
import pandas as pd
import numpy as np
import os
import warnings

from croprec import *

# def load_model(modelfile):
# 	loaded_model = pickle.load(open(modelfile, 'rb'))
# 	return loaded_model


    # title
html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Crop Recommendation  🌱 </h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

col1,col2  = st.columns([2,2])
    
with col1: 
        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''


with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen", 1,10000)
        P = st.number_input("Phosporus", 1,10000)
        K = st.number_input("Potassium", 1,10000)
        temp = st.number_input("Temperature",0.0,100000.0)
        humidity = st.number_input("Humidity in %", 0.0,100000.0)
        ph = st.number_input("Ph", 0.0,100000.0)
        rainfall = st.number_input("Rainfall in mm",0.0,100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            # loaded_model = load_model('model.pkl')
            # prediction = loaded_model.predict(single_pred)
            prediction = crop_prediction(single_pred)
            col1.write('''
		    ## Results 🔍 
		    ''')
            col1.success(f"{prediction.item().title()} is recommended by the model for this farm.")

