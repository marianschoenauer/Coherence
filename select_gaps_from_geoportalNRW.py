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
import seaborn as sns

R = "D:/coherence/friederike/"

# WMS layer 'windwurfschadflaechen_friederike' exported as GeoTiff, with
# 5x5 m resolution from: https://www.wms.nrw.de/umwelt/waldNRW
w = rio.open_rasterio(R+'fred5.tif')
w = w[0].where(w[0]!=0, np.nan)

# WMS layer 'baumartenklassifikation_nrw' exported as GeoTiff, with
# 5x5 m resolution from: https://www.wms.nrw.de/umwelt/waldNRW
b = rio.open_rasterio(R+'tree_species.tif')
# band signature checked to select canopy categories
beech = b[0].where(b[0] == 138, np.nan)
lhl = b[1].where(b[1] == 91, np.nan)
lhk = b[0].where(b[0] == 230, np.nan)
del b
# combined 3 types of broadleaved
broadleaved = xr.where(
    (~np.isnan(beech)) | (~np.isnan(lhl)) | (~np.isnan(lhk)),
    1,
    np.nan
)
del beech, lhl, lhk

# align the two rasters
w = w.rio.reproject_match(broadleaved)

# create intersection. as addition with nan remains nan
inter = w+broadleaved
inter = inter.drop_vars(['spatial_ref','band'])

# extract non-nan values and save coordinates
XY = []
for y in inter.y.values:
    xy=inter.sel(y = y).to_dataframe('p').dropna().set_index('y', append = True)
    if xy.size >0:
        XY.append(xy)

XY = pd.concat(XY).reset_index()

# transform coordinates to GeodataFrame
Points = gpd.points_from_xy(XY.x, XY.y).buffer(20).union_all()
Points = gpd.GeoDataFrame(geometry = [Points], crs = 3857).explode()

sns.histplot(Points.area)
# remove small areas
Points = Points.loc[Points.area>10000,:]

# export
R2 = "D:/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/shapefiles/"
Points.to_file(R2+'friederike_in_broadleaved.kml', driver = 'KML')
Points.to_file(R2+'friederike_in_broadleaved.gpkg')
