# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:25:27 2025

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

USER = "Marian"

if USER == "Marian":
    ROOT = "D:/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/"
else:
    ROOT = "C:/Users/Lika/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/"
    NC = "C:/Users/Lika/Desktop/py_coherence/NCs/"

AOIs = gpd.read_file(ROOT + "shapefiles/gaps.gpkg", layer = "AOIs_3857")\
    .set_index('AOI')
    
events = {
    'SLP':pd.Timestamp("2024-06-24"),
    'GER':pd.Timestamp("2018-01-18"),
    'ITA':pd.Timestamp("2018-10-29")
        }

for AOI, row in AOIs.iterrows():
    print(AOI)
    A = AOI[:3]
    
    event = np.datetime64(events.get(A))
    
    start = np.datetime64(event - pd.to_timedelta(4 if A == 'SLP' else 8, unit = 'W'))
    end = event  + pd.to_timedelta(4 if A == 'SLP' else 8, unit = 'W')

    # Sentinel-1
    s1_bs_full =  xr.load_dataset(ROOT +'satellite_data/S1_backscatter/cropped/'+AOI+'.nc',
                               engine = 'h5netcdf')
    s1_bs_full = s1_bs_full.rio.write_crs(s1_bs_full['spatial_ref'].attrs['crs_wkt'])

    # select time interval of interest. this is done to save file size
    s1_bs_full = s1_bs_full.sortby(['time']).sel({'time':slice(start, end)})

    CRS = s1_bs_full.spatial_ref.attrs['crs_wkt']
    s1_bs_full = s1_bs_full.rio.write_crs(CRS).rio.reproject(CRS)

    # Sentinel-2
    s2_full =  xr.load_dataset(ROOT + "satellite_data/s2/cropped/" + AOI + ".nc", \
                               engine = 'h5netcdf')
    s2_full = s2_full\
        .rio.write_crs(s2_full.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(s1_bs_full)
    
    #for i in s2_full.time[::5]:
    #    s2_full.sel({"time":i})['B4'].plot()
    #    plt.show()
    
    # select time interval of interest
    #s2_full = s2_full.sortby(['time']).sel({'time':slice(start, end)})
            
    # Coherence
    coherence_full =xr.load_dataset(ROOT + 'satellite_data/S1_coherence/cropped/'+AOI+'.nc',
                               engine = 'h5netcdf')

    coherence_full = coherence_full\
        .rio.write_crs(coherence_full.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(s1_bs_full)

    def coherence_baseline_days(days, dataset):
        """
        extract coherence layers based on baseline days.

        Parameters
        ----------
        days : INTEGER.
        dataset : XR.DATASET.

        Returns
        -------
        XR.DATASET.
        """

        name = "coh_" + str(days)
        coh_x = dataset.sel(pair=dataset.baseline_days == days)\
            .rename({'coherence':name})
        np.diff(coh_x.x)
        coh_x = coh_x.assign_coords(pair = coh_x.slave_time.values)
        coh_x = coh_x.rename({'pair':'time'})
        coh_x = coh_x.drop_vars(['baseline_days',
                                 'pair_name', 
                                 'master_time', 
                                 'slave_time'])
        return coh_x

    coh_12 = coherence_baseline_days(12, coherence_full)
    coh_24 = coherence_baseline_days(24, coherence_full)
    coh_36 = coherence_baseline_days(36, coherence_full)

    # merge and export
    conc = xr.merge([s1_bs_full,
                     s2_full,
                     coh_12, coh_24, coh_36
                      ],compat='no_conflicts',join='outer') 
    
    del s1_bs_full, s2_full, coh_12, coh_24, coh_36, coherence_full

    conc = conc.sortby(['x','y','time'])
    conc = conc.drop_vars('spatial_ref')
    conc = conc.drop_attrs()

    conc = conc.assign_coords(time = pd.to_timedelta(conc['time'] - event))
    
    try:
        conc = conc.drop_vars(['scene', 'source_file'])
    except:
        pass
    
    PATH_OUT = ROOT + 'satellite_data/merges_cropped/' +AOI+ ".nc"
    conc = conc.rio.write_crs(CRS).rio.reproject(CRS)
    conc.to_netcdf(PATH_OUT, engine= "h5netcdf")
    
    #conc['VV'].mean(dim = 'time').rio.to_raster('C:/Users/Marian Schonauer/Desktop/test/'+AOI+'.tif')

    del conc, event
