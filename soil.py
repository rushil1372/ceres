import geemap.foliumap as geemap
import ee
import datetime
import os
import streamlit as st
import time 

st.title('Soil Classification using Satellite Images')

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
l8 = l8.median()

Map.setCenter(78.9629, 20.5937, 8)
Map.addLayer(l8, {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3}, "Copernicus")
Map.to_streamlit()

from prediction import *

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0      

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        prediction = soil_predictor(os.path.join('uploaded',uploaded_file.name))
        st.text("Prediction: " + prediction)
        print(prediction)
        os.remove('uploaded/'+uploaded_file.name)

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

