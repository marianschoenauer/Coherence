# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:54:40 2025

@author: Marian Schonauer
"""

import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
ix = pd.IndexSlice

CRS = 5514

windthrows = gpd.read_file("D:/OneDrive - Mendelova univerzita v Brně/"
                           "Elizaveta Avoiani's files - Coherence_VI_Krtiny/"
                           "gis/windthrow.shp").to_crs(CRS)

# create empty multiindex MI
MI_used = pd.MultiIndex.from_arrays([[],[]],\
                                    names = ['x', 'y'])

for ID, row in windthrows.iterrows():

    # read in cluster (incl. buffer)
    PATH_CLUSTER = "D:/Coherence_VI_Krtiny/clusters/tp_plus_buffer_300m/"+\
        str(ID)+".nc"
    ds = xr.load_dataset(PATH_CLUSTER)\
        .rio.write_crs(CRS)

    # define true positives (tp), by clipping to extent of
    tp = ds.rio.clip(row[['geometry']])
    tp.to_netcdf(PATH_CLUSTER.replace("tp_plus_buffer_300m","tp"))

    # summary: counts of clusters
    tp_clusters = pd.Series(tp['cluster'].values.flatten()).value_counts()

    # define true negatives (tn), by inversed clipping
    tn = ds.rio.clip(windthrows.geometry.buffer(20), invert = True)
    
    tn.to_netcdf(PATH_CLUSTER.replace("tp_plus_buffer_300m","tn_plus_buffer_300m"))
    
    tn = tn\
        .to_dataframe()\
            .set_index('cluster', append = True)

    # remove the tn cells which were used already (MI_used. in first iteration,
    ## the index is empty, but it fills up)
    tn = tn.loc[~tn.index.isin(MI_used),:].copy()

    # blank multiindex with format as for tn
    MI_select_tn = pd.MultiIndex.from_arrays([[],[],[]], \
                                             names = tn.index.names)

    for clusterID, count in tp_clusters.items():

        # select each cluster-ID with "loc", and select the (maximum) number
        ## of counts with "iloc"
        tn_index = tn.loc[ix[:,:,clusterID],:].iloc[:count,:].index
        # append the multiindex
        MI_select_tn = MI_select_tn.append(tn_index)

        del tn_index

    # add MI_select_tn to the MI_used
    MI_used = MI_used.append(MI_select_tn.droplevel('cluster'))

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
                    .to_netcdf(PATH_CLUSTER\
                               .replace("tp_plus_buffer_300m","tn_selected"))
