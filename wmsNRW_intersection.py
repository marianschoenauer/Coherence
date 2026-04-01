# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:19:58 2026

@author: Marian Schonauer
"""
import xarray as xr
import rioxarray as rio
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

R = "D:/coherence/friederike/"

w = rio.open_rasterio(R+'fred5.tif')
w = w[0].where(w[0]!=0, np.nan)

b = rio.open_rasterio(R+'tree_species.tif')
beech = b[0].where(b[0] == 138, np.nan)
lhl = b[1].where(b[1] == 91, np.nan)
lhk = b[0].where(b[0] == 230, np.nan)

del b

LH = xr.where(
    (~np.isnan(beech)) | (~np.isnan(lhl)) | (~np.isnan(lhk)),
    1,
    np.nan
)
del beech, lhl, lhk

#plt.imshow(LH)

w = w.rio.reproject_match(LH)

inter = w+LH

inter = inter.drop_vars(['spatial_ref','band'])

XY = []

for y in inter.y.values:
    
    xy=inter.sel(y = y).to_dataframe('p').dropna().set_index('y', append = True)
    
    if xy.size >0:
        XY.append(xy)
        
XY = pd.concat(XY).reset_index()



Points = gpd.points_from_xy(XY.x, XY.y).buffer(20).union_all()
Points = gpd.GeoDataFrame(geometry = [Points], crs = 3857).explode()


R2 = "D:/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/shapefiles/"

Points.to_file(R2+'friederike_in_broadleaved.kml', driver = 'KML')
Points.to_file(R2+'friederike_in_broadleaved.gpkg')
