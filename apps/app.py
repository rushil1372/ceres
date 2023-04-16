import streamlit as st

import streamlit_book as stb
import geemap
from pathlib import Path

import os

# Set multipage
current_path = Path(__file__).parent.absolute()

# Streamit book properties
stb.set_book_config(menu_title="Main Menu",
                menu_icon="",
                options=[
                        "Soil Prediction",
                        "Crop Recommendation"
                        ],
                paths=[
                    current_path / "soil.py",
                    current_path / "crop.py"
                        ],
                icons=[
                        "earth",
                        "house"
                        ],
                )
    
