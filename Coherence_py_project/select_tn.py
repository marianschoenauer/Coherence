# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:54:40 2025

@author: Marian Schonauer
"""

import geopandas as gpd
import rasterio
import xarray as xr
import rioxarray as rio
import numpy as np
import pandas as pd
ix = pd.IndexSlice
import glob

windthrows = gpd.read_file("D:/OneDrive - Mendelova univerzita v Brně/Elizaveta Avoiani's files - Coherence_VI_Krtiny/gis/windthrow.shp").to_crs(CRS)


# read in clusters (incl. buffers) and loop though them
for path_AOI in glob.glob("D:/Coherence_VI_Krtiny/clusters/tp_plus_buffer_300m/*.nc"):
    ds = xr.load_dataset(path_AOI)\
        .rio.write_crs(CRS)
    
    # define true positives (tp), by clipping to extent of 
    tp = ds.rio.clip(windthrows.geometry)
    tp.to_netcdf(path_AOI.replace("tp_plus_buffer_300m","tp"))
    
    # summary: counts of clusters
    tp_clusters = pd.Series(tp['cluster'].values.flatten()).value_counts()
    
    # define true negatives, by inversed clipping
    tn = ds.rio.clip(AOIs.geometry.buffer(20), invert = True)\
        .to_dataframe()\
            .set_index('cluster', append = True)
            
    # blank multiindex with format as for tn  
    MI_select_tn = pd.MultiIndex.from_arrays([[],[],[]], \
                                             names = tn.index.names)
    
        
    for clusterID, count in tp_clusters.items():
        
        # select each cluster-ID with "loc", and select the (maximum) number of counts with "iloc"
        tn.loc[ix[:,:,clusterID],:].iloc[:count,:].index
        # append the multiindex
        MI_select_tn = MI_select_tn.append()

    # transform the multiindex to boolean series
    selected = tn.index.isin(MI_select_tn)
    
    tn = tn.reset_index('cluster')
    # all pixels not selected are overwritten with NA
    tn.loc[~selected,"cluster"] = np.nan
    
    # transform back to Dataset and export to netCDF
    xr.Dataset.from_dataframe(tn.drop(columns = ['spatial_ref']))\
        .transpose('y', 'x')\
            .rio.write_crs(CRS)\
                .rio.reproject(CRS)\
                    .to_netcdf(path_AOI.replace("tp_plus_buffer_300m","tn_selected"))