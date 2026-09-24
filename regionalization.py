# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:14:58 2025

@author: Marian Schonauer
"""

import gower
#pip install scikit-learn-extra
from sklearn_extra.cluster import KMedoids
import geopandas as gpd
import xarray as xr
import rioxarray as rio
import pandas as pd
ix = pd.IndexSlice
import numpy as np
import matplotlib.pyplot as plt

# %% select TN and TP
import os, glob
for path in glob.glob("D:/OneDrive - Mendelova univerzita v Brně/"
                      "Coherence_VI_Krtiny/clusters/TN/*"):
    os.remove(path)
for path in glob.glob("D:/OneDrive - Mendelova univerzita v Brně/"
                      "Coherence_VI_Krtiny/clusters/TP/*"):
    os.remove(path)

# %% forest mask

ROOT = ("D:/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/"
        "satellite_data/")

GAPS = ("D:/OneDrive - Mendelova univerzita v Brně/"
                           "Coherence_VI_Krtiny/"
                           "shapefiles/gaps.gpkg")

AOIs = gpd.read_file(GAPS, layer = "AOIs_3857").set_index('AOI')

gaps_conifer = gpd.read_file(GAPS, layer = "gap_in_conifers")

CRS = 3857

for AOI, row, in AOIs.iterrows():#'USA',
    print(AOI)
    A = AOI[:3]
    
    fm = xr.load_dataset(ROOT + 'DynWorld/cropped/'+AOI+'.nc', engine = 'h5netcdf')
    CRS_DS = fm['spatial_ref'].attrs['crs_wkt']
    fm = fm.rio.write_crs(CRS_DS).rio.reproject(CRS)
        
    forestMask = fm.where(fm.label == 1)['label'].squeeze()
        
    gaps_conifer = gaps_conifer.to_crs(CRS)
    
    forestMask = forestMask.rio.clip(gaps_conifer.geometry, invert = True)
                    
    if A == 'SLP':
        all_gaps = gpd.read_file(GAPS, layer = "canopy_diffs")\
            .to_crs(CRS)
            
    if A == 'ITA':
        all_gaps = gpd.read_file(GAPS, layer = "Italy_windthrows")\
            .to_crs(CRS)
    
    if A == 'GER':      
        all_gaps = gpd.read_file(GAPS, layer = "Germany_Friederike")\
            .to_crs(CRS)
    
    #  select TN and TP
    
    ## exclude small areas
    one_pixel_coherence = 20*2
    
    larger_gaps = all_gaps[all_gaps.area > one_pixel_coherence].copy()
    small_gaps = all_gaps[all_gaps.area < one_pixel_coherence].copy()
      
    ## make one geometry
    
    LARGER_GAPS = larger_gaps.to_crs(forestMask.rio.crs).union_all()
    
    if small_gaps.shape[0] > 1:
        SMALL_GAPS = small_gaps.union_all()
    
    # Loop through AOIsgit

    TP = forestMask.rio.clip([LARGER_GAPS])
    TN = forestMask.rio.clip([LARGER_GAPS], invert = True)

    
    if small_gaps.shape[0] > 1:
        TN = TN.rio.clip([SMALL_GAPS], invert = True)

    fig, ax = plt.subplots(2,1)
    
    TP.plot(ax = ax[0])
    TN.plot(ax = ax[1])
    ax[0].set_title(AOI)
    plt.show()       

    TP.rio.to_raster("D:/OneDrive - Mendelova univerzita v Brně/"
               "Coherence_VI_Krtiny/clusters/TP/"+
                      str(AOI)+".tif")
    
    TN.rio.to_raster("D:/OneDrive - Mendelova univerzita v Brně/"
               "Coherence_VI_Krtiny/clusters/TN/"+
                      str(AOI)+".tif")
