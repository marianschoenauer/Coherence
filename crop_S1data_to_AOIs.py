# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:17:37 2026

@author: Marian Schonauer
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rio
import matplotlib.pyplot as plt

AOIs = gpd.read_file("D:/OneDrive - Mendelova univerzita v Brně/"
                           "Coherence_VI_Krtiny/"
                           "shapefiles/gaps.gpkg", 
                           layer = "AOIs_individual_gaps_3035")
AOIs['AOI'] = AOIs['Country'] + "_" + AOIs['ID_No'].astype(str)
AOIs = AOIs.set_index('AOI')
AOIs.loc[:,'geometry'] = AOIs.geometry.buffer(250)

CRS = AOIs.crs

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

for path_, library in [('S1_backscatter/',  backscatter_names),
                             ('S1_coherence/', coherence_names)]:
    print(path_, '\n')

    for A  in ['SLP','ITA','GER']:
        print(A)

        ds = xr.load_dataset(ROOT + path_ + library.get(A), engine = 'h5netcdf')
        crs_ds = ds['spatial_ref'].attrs['crs_wkt']
        ds = ds.rio.write_crs(crs_ds).rio.reproject(CRS).sortby(['x', 'y'])

        aoi = AOIs.loc[AOIs['Country'] == A,:].copy()

        for AOI, row in aoi.iterrows():
            print(AOI)
            xmin, ymin, xmax, ymax = row['geometry'].bounds

            ds_crop = ds.sel({'x':slice(xmin, xmax), 'y':slice(ymin, ymax)})
            
            if path_ == "S1_backscatter/":
                ds_crop.isel(time = 1)['VV'].plot()
                plt.show()
            else:
                ds_crop.isel(pair = 1)['coherence'].plot()
                plt.show()

            ds_crop.to_netcdf(ROOT + path_ + 'cropped/' + AOI + '.nc',
                engine = 'h5netcdf')