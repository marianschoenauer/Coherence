# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:25:27 2025

@author: Marian Schonauer
"""
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import glob
#import geopandas as gpd
import xarray as xr
import rioxarray as rio
import pandas as pd
ix = pd.IndexSlice
import seaborn as sns

ROOT = ("D:/OneDrive - Mendelova univerzita v Brně/"
           "Coherence_VI_Krtiny/satellite_data/")

# raster data

AOIs = gpd.read_file("D:/OneDrive - Mendelova univerzita v Brně/"
                           "Coherence_VI_Krtiny/"
                           "shapefiles/gaps.gpkg", layer = "AOIs_3857")

#   if A == 'USA':        event = np.datetime64("2018-03-01")

for A, name_s1_backscatter, name_s1_coherence, event in \
    [('GER', 
      'GER_backscatter_dB.nc', 
      'Fridrieke_coherence_stack_GER_test_0_clean_v2.nc',
      np.datetime64("2018-01-18")),
     
     ('SLP',
      'SLP_backscatter_stack_10m_2024_dB_32633_v2.nc',
      'SLP_coherence_stack_10m_2024_32633.nc',
      np.datetime64("2024-06-21")),
     
     ('ITA', 
      'ITA_backscatter_dB.nc', 
      'ITA_coherence_stack_INT40_40m_32633_12day_pairs.nc',
      np.datetime64("2018-10-29"))
     ]: 
        
        
    start = np.datetime64(event - pd.to_timedelta(4, unit = 'W'))
    end = event + pd.to_timedelta(8 if A == 'GER' else 4, unit = 'W')
    
    print(A)
    
    # Sentinel-1
    
    s1_full =  xr.load_dataset(ROOT +'S1_backscatter/'+name_s1_backscatter, 
                               engine = 'h5netcdf')  
    
    s1_full = s1_full.sortby(['time']).sel({'time':slice(start, end)})
    
    CRS = s1_full.spatial_ref.attrs['crs_wkt']
    
    s1_full = s1_full\
        .rio.write_crs(CRS)


    '''
    
    A, name_s1_backscatter =  'GER', 'Friedrike_backscatter_32632.nc'
    
    s1_full =  xr.load_dataset(ROOT +'S1_backscatter/'+name_s1_backscatter, 
                               engine = 'h5netcdf')
    
    s1_full = s1_full[['VV','VH']]
    
    s1_full['VV'] = 10*np.log10(s1_full['VV'])
    s1_full['VH'] = 10*np.log10(s1_full['VH'])  
    
    CRS = s1_full.spatial_ref.attrs['crs_wkt']
    
    s1_full = s1_full\
        .rio.write_crs(CRS)
    
    s1_full.to_netcdf(ROOT +'S1_backscatter/GER_backscatter_dB.nc', engine = 'h5netcdf')
    
    
    # ITA
    A, name_s1_backscatter =  'ITA', 'ITA_backscatter_stack_RTC10_10m_32633_12day_pairs.nc'
    
    s1_full =  xr.load_dataset(ROOT +'S1_backscatter/'+name_s1_backscatter, 
                               engine = 'h5netcdf')
    
    s1_full = s1_full[['gamma0_VV','gamma0_VH']].rename({'gamma0_VV':'VV','gamma0_VH':'VH'})
    
    s1_full['VV'] = 10*np.log10(s1_full['VV'])
    s1_full['VH'] = 10*np.log10(s1_full['VH'])  
    
    CRS = s1_full.spatial_ref.attrs['crs_wkt']
    
    s1_full = s1_full\
        .rio.write_crs(CRS)
    
    s1_full.to_netcdf(ROOT +'S1_backscatter/ITA_backscatter_dB.nc', engine = 'h5netcdf')

    '''

    AOI = AOIs.loc[AOIs.AOI.str.startswith(A),:].copy()
    xmin, ymin, xmax, ymax = AOI.to_crs(CRS).total_bounds
    
    s1_full = s1_full.sel({'x':slice(xmin, xmax), 'y':slice(ymax,ymin)})
    
    del AOI, xmin, ymin, xmax, ymax
    '''
    s1_full =  xr.load_dataset(ROOT+'s1/'+A+'.nc', 
                               engine = 'h5netcdf')
    
    CRS = s1_full.spatial_ref.attrs['crs_wkt']
    
    s1_full = s1_full\
        .rio.write_crs(CRS)
    np.diff(s1_full.x)
    
    xmin, ymin, xmax, ymax = AOIs.to_crs(CRS).total_bounds
    
    #s1_full = s1_full.sel({'x':slice(xmin, xmax), 'y':slice(ymax,ymin)})   
    
    # Landsat
    s2_full =  xr.load_dataset(ROOT + "landsat/"
                    +A+".nc", engine = 'h5netcdf')
    
    np.diff(s2_full.x)
    s2_full = s2_full\
        .rio.write_crs(s2_full.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(s1_full)
            
    s2_full["NDVI"]  = (s2_full["SR_B5"] - s2_full["SR_B4"]) / (s2_full["SR_B5"] + s2_full["SR_B4"])
    '''

    # Sentinel-2
    s2_full =  xr.load_dataset(ROOT + "s2/"
                    +A+".nc", engine = 'h5netcdf')
    
    s2_full = s2_full.sortby(['time']).sel({'time':slice(start, end)})
    
    s2_full = s2_full\
        .rio.write_crs(s2_full.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(s1_full)

    #s2_full["NDVI"]  = (s2_full["B8"] - s2_full["B4"]) / (s2_full["B8"] + s2_full["B4"])
    
    # Coherence
    
    coherence_full =xr.load_dataset(ROOT + 'S1_coherence/'+ name_s1_coherence, 
                               engine = 'h5netcdf')
    
    s2_full = s2_full.sortby(['time']).sel({'time':slice(start, end)})

    coherence_full = coherence_full\
        .rio.write_crs(coherence_full.spatial_ref.attrs['spatial_ref'])\
            .rio.reproject_match(s1_full)

    def coherence_baseline_days(days, Dataset):
        NAME = "coh_" + str(days)
        coh_X = Dataset.sel(pair=Dataset.baseline_days == days)\
            .rename({'coherence':NAME})
        np.diff(coh_X.x)
        coh_X = coh_X.assign_coords(pair = coh_X.slave_time.values)
        coh_X = coh_X.rename({'pair':'time'})
        coh_X = coh_X.drop_vars(['baseline_days', 'pair_name', 'master_time', 'slave_time'])
        return coh_X

    coh_12 = coherence_baseline_days(12, coherence_full)
    #coh_24 = coherence_baseline_days(24, coherence_full)
    #coh_36 = coherence_baseline_days(36, coherence_full)

    # merge and export
    conc = xr.merge([s1_full, 
                     s2_full,
                     coh_12, #coh_24, coh_36
                      ])
    
    CRS_s1 = s1_full.spatial_ref.attrs['crs_wkt']
    
    del s1_full, s2_full, coh_12, coherence_full
    
    conc = conc.sortby(['x','y','time'])
    
    conc = conc.drop_vars('spatial_ref')
    conc = conc.rio.write_crs(CRS_s1)\
        .rio.reproject(CRS_s1)

    if A == 'GER':
        conc = conc.drop(['scene', 'source_file'])

    #conc['WI'].isel(time = 0).rio.to_raster(ROOT + 'VV_usa.tif')

    conc = conc.assign_coords(time = pd.to_timedelta(conc['time'] - event))
    del event
    

    conc = conc.drop_attrs()
    conc = conc.rio.write_crs(CRS)

    print(ROOT +A+ "_conc.nc")
    conc.to_netcdf(ROOT +A+ "_conc.nc", engine= "h5netcdf")

    del conc
