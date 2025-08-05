#!/usr/bin/env python
# coding: utf-8

# In[1]:


# generate_cd_vector.py

import pandas as pd

def generate_cd_vector(keystrokes_df, mouse_move_df, mouse_click_df, label):
    cd_vector = {}

    cd_vector["MeanHoldTime"] = keystrokes_df["HoldTime"].mean()
    cd_vector["StdHoldTime"] = keystrokes_df["HoldTime"].std()

    cd_vector["MeanSpeed"] = mouse_move_df["Speed"].mean()
    cd_vector["StdSpeed"] = mouse_move_df["Speed"].std()
    cd_vector["MeanAcceleration"] = mouse_move_df["Acceleration"].mean()
    cd_vector["StdAcceleration"] = mouse_move_df["Acceleration"].std()

    cd_vector["MeanClickDuration"] = mouse_click_df["ClickDuration"].mean()
    cd_vector["StdClickDuration"] = mouse_click_df["ClickDuration"].std()

    cd_vector["KeystrokeCount"] = len(keystrokes_df)
    cd_vector["MouseMoveCount"] = len(mouse_move_df)
    cd_vector["MouseClickCount"] = len(mouse_click_df)

    cd_vector["Label"] = label
    cd_vector["Timestamp"] = None  # Will be added in main_capture.py

    return pd.DataFrame([cd_vector])
