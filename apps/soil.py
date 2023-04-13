import geemap.foliumap as geemap
import ee

import datetime

import streamlit as st


st.title('Soil Classification using Satellite Images')

Map = geemap.Map(zoom=1)

def maskS2clouds(image):
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloudBitMask = 1 << 10
  cirrusBitMask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))

  return image.updateMask(mask).divide(10000)


l8 = ee.ImageCollection('COPERNICUS/S2').filterDate('2023-01-01', '2023-03-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(maskS2clouds)

Map.setCenter(78.9629, 20.5937, 8)
Map.addLayer(l8.median(), {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3}, "Copernicus")
Map.to_streamlit()