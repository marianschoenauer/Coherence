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
           "Coherence_VI_Krtiny/")

# %% raster data

windthrows = gpd.read_file("D:/OneDrive - Mendelova univerzita v Brně/"
                           "Coherence_VI_Krtiny/"
                           "shapefiles/gaps.gpkg", layer = "AOIs")



# Sentinel-1
s1_full =  xr.load_dataset(ROOT + "coherence_backscatter_data/netCDF/"
                           "backscatter_stack_Krtiny_10m_2024_dB_32633.nc", 
                           engine = 'h5netcdf')\
    .rename({'gamma0_VV_dB':'VV', 'gamma0_VH_dB':'VH'})\
        .drop_vars(['gamma0_VV','gamma0_VH'])
CRS = s1_full.spatial_ref.attrs['crs_wkt']
s1_full = s1_full.assign_coords(x = (s1_full.x.values - 30))

s1_full = s1_full\
    .rio.write_crs(CRS)
np.diff(s1_full.x)

xmin, ymin, xmax, ymax = windthrows.to_crs(CRS).total_bounds

s1_full = s1_full.sel({'x':slice(xmin, xmax), 'y':slice(ymax,ymin)})

# Sentinel-2
s2_full =  xr.load_dataset(ROOT + "s2/"
                "0.nc", engine = 'h5netcdf')
np.diff(s2_full.x)
s2_full = s2_full\
    .rio.write_crs(s2_full.spatial_ref.attrs['crs_wkt'])\
        .rio.reproject_match(s1_full)
        
s2_full["NDVI"]  = (s2_full["B8"] - s2_full["B4"]) / (s2_full["B8"] + s2_full["B4"])

# Coherence

coherence_full =xr.load_dataset(ROOT + "coherence_backscatter_data/netCDF/"
                           "coherence_stack_Krtiny_10m_2024_32633.nc", 
                           engine = 'h5netcdf')

#coherence_full = coherence_full.assign_coords(x = (coherence_full.x.values - 30))
  

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
coh_24 = coherence_baseline_days(24, coherence_full)
coh_36 = coherence_baseline_days(36, coherence_full)


conc = xr.merge([s1_full, 
                  s2_full,
                  coh_12,coh_24,coh_36
                  ]).sortby(['x','y','time'])


conc['Rc'] = conc['VH'] - conc['VV']

conc = conc.drop_vars('spatial_ref')
conc = conc.rio.write_crs(s1_full.spatial_ref.attrs['crs_wkt'])\
    .rio.reproject(s1_full.spatial_ref.attrs['crs_wkt'])

conc = conc.assign_coords(time = pd.to_timedelta(conc['time'] - np.datetime64("2024-06-21")))
del coh_12, coh_24, coh_36, coherence_full, s1_full, s2_full



conc = conc.drop_attrs()
conc = conc.rio.write_crs(CRS)

conc.to_netcdf(ROOT + "conc.nc", engine= "h5netcdf")
