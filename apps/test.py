import ee
import os
import datetime
import fiona
import geopandas as gpd
import folium
import streamlit as st
import geemap.colormaps as cm
import geemap.foliumap as geemap
from datetime import date


@st.cache_data
def uploaded_file_to_gdf(data):
    import tempfile
    import os
    import uuid

    _, file_extension = os.path.splitext(data.name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    with open(file_path, "wb") as file:
        file.write(data.getbuffer())

    if file_path.lower().endswith(".kml"):
        fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(file_path, driver="KML")
    else:
        gdf = gpd.read_file(file_path)

    return gdf


def app():

    today = date.today()

    st.title("Create Timelapse")

    row1_col1, row1_col2 = st.columns([2, 1])

    if st.session_state.get("zoom_level") is None:
        st.session_state["zoom_level"] = 4

    st.session_state["ee_asset_id"] = None
    st.session_state["bands"] = None
    st.session_state["palette"] = None
    st.session_state["vis_params"] = None

    with row1_col1:
        m = geemap.Map(
            basemap="HYBRID",
            plugin_Draw=True,
            Draw_export=True,
            locate_control=True,
            plugin_LatLngPopup=False,
        )

    with row1_col2:

        collection = st.selectbox(
            "Select a satellite image collection: ",
            [
                "Landsat TM-ETM-OLI Surface Reflectance",
                "Sentinel-2 MSI Surface Reflectance",
            ],
            index=1,
        )

        if collection in [
            "Landsat TM-ETM-OLI Surface Reflectance",
            "Sentinel-2 MSI Surface Reflectance",
        ]:
            roi_options = ["Uploaded GeoJSON"] + list(landsat_rois.keys())


    with row1_col1:

        data = st.file_uploader(
            "Upload a GeoJSON file to use as an ROI. Customize timelapse parameters and then click the Submit button 😇👇",
            type=["geojson", "kml", "zip"],
        )

        crs = "epsg:4326"
        if sample_roi == "Uploaded GeoJSON":
            if data is None:
                if collection in [
                    "Geostationary Operational Environmental Satellites (GOES)",
                    "USDA National Agriculture Imagery Program (NAIP)",
                ] and (not keyword):
                    m.set_center(-100, 40, 3)
                # else:
                #     m.set_center(4.20, 18.63, zoom=2)
        else:
            if collection in [
                "Landsat TM-ETM-OLI Surface Reflectance",
                "Sentinel-2 MSI Surface Reflectance",
            ]:
                gdf = gpd.GeoDataFrame(
                    index=[0], crs=crs, geometry=[landsat_rois[sample_roi]]
                )
            elif (
                collection
                == "Geostationary Operational Environmental Satellites (GOES)"
            ):
                gdf = gpd.GeoDataFrame(
                    index=[0], crs=crs, geometry=[goes_rois[sample_roi]["region"]]
                )
            elif collection == "MODIS Vegetation Indices (NDVI/EVI) 16-Day Global 1km":
                gdf = gpd.GeoDataFrame(
                    index=[0], crs=crs, geometry=[modis_rois[sample_roi]]
                )

        if sample_roi != "Uploaded GeoJSON":

            if collection in [
                "Landsat TM-ETM-OLI Surface Reflectance",
                "Sentinel-2 MSI Surface Reflectance",
            ]:
                gdf = gpd.GeoDataFrame(
                    index=[0], crs=crs, geometry=[landsat_rois[sample_roi]]
                )
            elif (
                collection
                == "Geostationary Operational Environmental Satellites (GOES)"
            ):
                gdf = gpd.GeoDataFrame(
                    index=[0], crs=crs, geometry=[goes_rois[sample_roi]["region"]]
                )
            st.session_state["roi"] = geemap.gdf_to_ee(gdf, geodesic=False)
            m.add_gdf(gdf, "ROI")

        elif data:
            gdf = uploaded_file_to_gdf(data)
            st.session_state["roi"] = geemap.gdf_to_ee(gdf, geodesic=False)
            m.add_gdf(gdf, "ROI")

        m.to_streamlit(height=600)
