# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:45:28 2025

@author: Marian Schonauer
"""

import glob
import geopandas as gpd
import xarray as xr
import rioxarray as rio
import pandas as pd
ix = pd.IndexSlice
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



ROOT = ("D:/OneDrive - Mendelova univerzita v Brně/"
           "Coherence_VI_Krtiny/")



ds = rio.open_rasterio(ROOT + "canopy_height/Lidar_P95/diff_summer-winter2024.tif", masked = True)
CRS = ds.spatial_ref.attrs['crs_wkt']

xs = np.abs(np.diff(ds.x)).max()
ys = np.abs(np.diff(ds.y)).max()

s = np.max([xs,ys])

df =  ds.to_dataframe(name = 'diff').dropna().droplevel(0)

df = df.loc[(df['diff']<-5) & (df['diff']>-50),:].reset_index()



sp = gpd.GeoDataFrame(df['diff'], geometry=gpd.points_from_xy(df.x, df.y, crs = CRS))

sp['geometry'] = sp.geometry.buffer((s/2)+0.01, cap_style = 'square')

Union = sp.union_all()

sp = gpd.GeoDataFrame(geometry = [Union],crs = CRS)

sp = sp.explode()

sp.to_file(ROOT + "shapefiles/" + 'gaps.gpkg', layer = "canopy_diffs")


