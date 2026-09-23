# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:25:27 2025

@author: Marian Schonauer
"""

#import glob
import geopandas as gpd
import numpy as np
#import matplotlib.pyplot as plt
#import geopandas as gpd
import xarray as xr
import rioxarray as rio
import pandas as pd
ix = pd.IndexSlice
#import seaborn as sns

USER = "Lika"

if USER == "Marian":
    ROOT = "D:/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/"
else:
    ROOT = "C:/Users/Lika/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/"
    NC = "C:/Users/Lika/Desktop/py_coherence/NCs/"

AOIs = gpd.read_file(ROOT + "shapefiles/gaps.gpkg", layer = "AOIs_3857")

#   if A == 'USA':        event = np.datetime64("2018-03-01")

for A, name_s1_backscatter, name_s1_coherence, event in \
    [
     ('SLP',
        'SLP_backscatter_stack_10m_2024_dB_32633_v2.nc',
        'SLP_coherence_stack_10m_2024_32633.nc',
        np.datetime64("2024-06-21")),
     ('GER',
        'GER_backscatter_dB.nc', 
        'Fridrieke_coherence_stack_GER_test_0_clean_v2.nc',
        np.datetime64("2018-01-18")),
     ('ITA',
        'ITA_backscatter_dB.nc', 
        'ITA_coherence_stack_INT40_40m_32633_12day_pairs.nc',
        np.datetime64("2018-10-29"))
     ]:

    start = np.datetime64(event - pd.to_timedelta(8 , unit = 'W'))
    end = event + pd.to_timedelta(8 , unit = 'W')

    print(A)

    # Sentinel-1
    s1_bs_full =  xr.load_dataset(ROOT +'satellite_data/S1_backscatter/'+name_s1_backscatter,
                               engine = 'h5netcdf')
    s1_bs_full = s1_bs_full.rio.write_crs(s1_bs_full['spatial_ref'].attrs['crs_wkt'])

    # select time interval of interest. this is done to save file size
    s1_bs_full = s1_bs_full.sortby(['time']).sel({'time':slice(start, end)})

    CRS = s1_bs_full.spatial_ref.attrs['crs_wkt']
    s1_bs_full = s1_bs_full.rio.write_crs(CRS).rio.reproject(CRS)

    # crop to AOIs total bounds
    AOI = AOIs.loc[AOIs.AOI.str.startswith(A),:].copy()
    xmin, ymin, xmax, ymax = AOI.to_crs(CRS).total_bounds
    s1_bs_full = s1_bs_full.sel({'x':slice(xmin, xmax), 'y':slice(ymax,ymin)})

    del AOI, xmin, ymin, xmax, ymax

    # Sentinel-2
    s2_full =  xr.load_dataset(ROOT + "satellite_data/s2/" + A + ".nc", \
                               engine = 'h5netcdf')
    s2_full = s2_full.rio.write_crs(s2_full['spatial_ref'].attrs['crs_wkt'])

    # select time interval of interest
    #s2_full = s2_full.sortby(['time']).sel({'time':slice(start, end)})

    s2_full = s2_full\
        .rio.write_crs(s2_full.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(s1_bs_full)
            
    # Coherence
    coherence_full =xr.load_dataset(ROOT + 'satellite_data/S1_coherence/'+ name_s1_coherence,
                               engine = 'h5netcdf')
    coherence_full = coherence_full.rio.write_crs(coherence_full['spatial_ref'].attrs['crs_wkt'])

    # select time interval of interest not applicable

    coherence_full = coherence_full\
        .rio.write_crs(coherence_full.spatial_ref.attrs['spatial_ref'])\
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
    if A == 'GER':
        conc = conc.drop_vars(['scene', 'source_file'])

    print(NC + 'satellite_data/' +A+ "_conc.nc")

    conc = conc.rio.write_crs(CRS).rio.reproject(CRS)
    conc.to_netcdf(NC +A+ "_conc.nc", engine= "h5netcdf")

    del conc, event
