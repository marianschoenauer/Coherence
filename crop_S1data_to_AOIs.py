# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:17:37 2026

@author: Marian Schonauer
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:16:52 2026

@author: Marian Schonauer
"""

#import glob
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
#import geopandas as gpd
import xarray as xr
import rioxarray as rio
import pandas as pd
ix = pd.IndexSlice
#import seaborn as sns

geoms = gpd.read_file("D:/OneDrive - Mendelova univerzita v Brně/"
                           "Coherence_VI_Krtiny/"
                           "shapefiles/gaps.gpkg", layer = "AOIs_3857")

ROOT = 'D:/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/satellite_data/'


backscatter_names = {
    'SLP':'SLP_backscatter_stack_10m_2024_dB_32633_v2.nc',
    'GER':'GER_backscatter_dB.nc',
    'ITA':'ITA_backscatter_dB.nc'    
    }

coherence_names = {
    'SLP':'SLP_coherence_stack_10m_2024_32633.nc',
    'GER':'Fridrieke_coherence_stack_GER_test_0_clean_v2.nc',
    'ITA':'ITA_coherence_stack_INT40_40m_32633_12day_pairs.nc'    
    }

for A  in ['SLP','ITA','GER']:
    
    print(A)
   
    ds = xr.load_dataset(ROOT + 'S1_backscatter/' + backscatter_names.get(A), engine = 'h5netcdf')\
        .sortby(['x','y'])
        
    crs_ds = ds['spatial_ref'].attrs['crs_wkt']
    
    aoi = geoms.loc[geoms.AOI.str.startswith(A),:].to_crs(crs_ds)

    aoi.loc[:,'geometry'] = aoi.geometry.buffer(250)
    aoi = aoi.set_index('AOI')  

    for AOI, row in aoi.iterrows():
        print(AOI)
        xmin, ymin, xmax, ymax = row['geometry'].bounds
        
        ds_crop = ds.sel({'x':slice(xmin, xmax), 'y':slice(ymin, ymax)})
        
        ds_crop.to_netcdf(ROOT + 'S1_backscatter/cropped/' + AOI + '.nc',
            engine = 'h5netcdf')

for A  in ['SLP','GER','ITA']:
    
    print(A)
   
    ds = xr.load_dataset(ROOT + 'S1_coherence/' + coherence_names.get(A), engine = 'h5netcdf')\
        .sortby(['x','y'])
        
    crs_ds = ds['spatial_ref'].attrs['crs_wkt']
    
    aoi = geoms.loc[geoms.AOI.str.startswith(A),:].to_crs(crs_ds)

    aoi.loc[:,'geometry'] = aoi.geometry.buffer(250)
    aoi = aoi.set_index('AOI')  

    for AOI, row in aoi.iterrows():
        print(AOI)
        xmin, ymin, xmax, ymax = row['geometry'].bounds
        
        ds_crop = ds.sel({'x':slice(xmin, xmax), 'y':slice(ymin, ymax)})
        
        ds_crop.to_netcdf(ROOT + 'S1_coherence/cropped/' + AOI + '.nc',
            engine = 'h5netcdf')