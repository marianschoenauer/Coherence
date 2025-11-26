# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:25:27 2025

@author: Marian Schonauer
"""
import glob
import gower
#pip install scikit-learn-extra
from sklearn_extra.cluster import KMedoids
import geopandas as gpd
import xarray as xr
import rioxarray as rio
import pandas as pd
ix = pd.IndexSlice
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
CRS =5514

ROOT = ("D:/OneDrive - Mendelova univerzita v Brně/"
"IGA Team - UFE Wind Throw - General/GIS/")


# %% raster data

nc_target = rio.open_rasterio(ROOT + "forest/canopy/"
                                "TreeSpecies_PlanetScope.tif")\
    .squeeze()\
        .rio.reproject(CRS)\
            .rename("PlanSco")\
                .sel(x = slice(-590390,-582317), y = slice(-1148377,-1149648))

"""backscatter =xr.load_dataset("D:/Coherence_VI_Krtiny/coherence_backscatter_data/"
                "netCDF/backscatter_stack_Krtiny_10m_32633.nc")
backscatter = backscatter\
    .rio.write_crs(backscatter.crs_wkt)\
        .rio.reproject_match(nc_target)\
            .groupby(backscatter.time.dt.strftime('%Y%m%d')).mean()"""
        
coherence =xr.load_dataset("D:/Coherence_VI_Krtiny/coherence_backscatter_data/"
                "netCDF/coherence_stack_Krtiny_40m_32633.nc")
coherence = coherence\
    .rio.write_crs(coherence.spatial_ref.attrs['spatial_ref'])\
        .rio.reproject_match(nc_target)\
            .groupby(coherence.time.dt.strftime('%Y%m%d')).mean()
            
s1 =  xr.load_dataset("D:/Coherence_VI_Krtiny/s1/"
                "0.nc")
s1 = s1.rio.write_crs(s1.spatial_ref.attrs['crs_wkt'])\
        .rio.reproject_match(nc_target)\
            .groupby(s1.time.dt.strftime('%Y%m%d')).mean()
            
s2 =  xr.load_dataset("D:/Coherence_VI_Krtiny/s2/"
                "0.nc")
s2 = s2.rio.write_crs(s2.spatial_ref.attrs['crs_wkt'])\
        .rio.reproject_match(nc_target)\
            .groupby(s2.time.dt.strftime('%Y%m%d')).mean()

df = xr.merge([coherence, s1,s2])\
    .to_dataframe()\
        .astype(float)\
                .sort_index()
                
df.loc[:,'NDVI'] = (df.B8 - df.B4) / (df.B8 + df.B4)
df.loc[:,'NDWI'] = (df.B3 - df.B8A) / (df.B3 + df.B8A)
df.loc[:,'Rc'] = df.VH / df.VV
df.loc[:,'RVI'] = (4 * df.VH) / (df.VV + df.VH)
df.loc[:,'RNDVI'] = (df.VH - df.VV) / (df.VH + df.VV)        
                
del coherence, s2, s1
# %% read tiles

Df = []

for PATH in glob.glob("D:/Coherence_VI_Krtiny/clusters/tp/*.nc"):
    
    tp = xr.load_dataset(PATH)
    tp = tp.rio.write_crs(tp.spatial_ref.attrs['spatial_ref'])\
            .rio.reproject_match(nc_target)\
                .to_dataframe()\
                    .dropna()
    
    tn = xr.load_dataset(PATH.replace("tp","tn_selected"))
    tn = tn.rio.write_crs(tn.spatial_ref.attrs['spatial_ref'])\
            .rio.reproject_match(nc_target)\
                .to_dataframe()\
                    .dropna()
       
    tp_sat = df.loc[df.index.isin(tp.index),:].assign(value = 'tp', 
                  ID = PATH.replace("D:/Coherence_VI_Krtiny/clusters/tp\\",""))
    tn_sat = df.loc[df.index.isin(tn.index),:].assign(value = 'tn', 
                  ID = PATH.replace("D:/Coherence_VI_Krtiny/clusters/tp\\",""))
    
    Df.append(tp_sat)
    Df.append(tn_sat)
    
    del tp, tn, tp_sat, tn_sat
    
Df = pd.concat(Df)

Df['time'] = pd.to_datetime(Df.reset_index()['strftime']).values


Df.columns
# %%
fig, ax = plt.subplot_mosaic([['coherence'],['RNDVI'], ['NDVI']])

for var in ['coherence', 'RNDVI', 'NDVI']:
    sns.lineplot(Df.reset_index(), x = 'time', y = var, hue = 'value', ax = ax[var])
    
    ax[var].tick_params(axis='x', rotation=90)
    ax[var].axvline(np.datetime64("2024-06-21"))
