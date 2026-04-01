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
# %%% SLP
CRS =3857
ROOT = ("D:/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/"
        "satellite_data/")

AOIs = gpd.read_file("D:/OneDrive - Mendelova univerzita v Brně/"
                           "Coherence_VI_Krtiny/"
                           "shapefiles/gaps.gpkg", layer = "AOIs")\
    .to_crs(CRS)\
        .set_index('AOI')
        
A = 'SLP'#

for A in ['SLP', 'USA']:
       
    
    fm = xr.load_dataset(ROOT + 'DynWorld/'+A+'.nc', engine = 'h5netcdf')\
        .rio.write_crs(4326)\
            .rio.reproject(CRS)
            
    fm['label'].plot()
    plt.show()
        
    forestMask = fm.where(fm.label == 1)['label']\
        .squeeze()
        
    forestMask.rio.to_raster(ROOT+'DynWorld_'+A+'_1.tif')
        
    if A == 'SLP':
        all_gaps = gpd.read_file("D:/OneDrive - Mendelova univerzita v Brně/"
                                   "Coherence_VI_Krtiny/"
                                   "shapefiles/gaps.gpkg", layer = "canopy_diffs")\
            .to_crs(CRS)
            
    if A == 'USA':
        all_gaps = gpd.read_file("D:/OneDrive - Mendelova univerzita v Brně/"
                                   "Coherence_VI_Krtiny/"
                                   "shapefiles/gaps.gpkg", layer = "usa")\
            .to_crs(CRS)
            
    
    #1 	#397d49 	trees
    
    #  select TN and TP
    
    ## exclude small areas
    one_pixel_coherence = 10*10
    
    larger_gaps = all_gaps[all_gaps.area > one_pixel_coherence*2].copy()
    small_gaps = all_gaps[all_gaps.area < one_pixel_coherence*2].copy()
    
    ## make one geometry
    
    LARGER_GAPS = larger_gaps.union_all()
    
    if small_gaps.shape[0] > 1:
        SMALL_GAPS = small_gaps.union_all()
    
    # Loop through AOIsgit
    
    
    for Area, row in AOIs.loc[AOIs.index.str.startswith(A),:].iterrows():
    
        print(f'Processing area {Area}...')
          
        ds_AOI = forestMask.rio.clip(row)
    
        TP = ds_AOI.rio.clip([LARGER_GAPS])
        TN = ds_AOI.rio.clip([LARGER_GAPS], invert = True)
        if small_gaps.shape[0] > 1:
            TN = TN.rio.clip([SMALL_GAPS], invert = True)
            
        fig, ax = plt.subplots(2,1)
        
        TP.plot(ax = ax[0])
        TN.plot(ax = ax[1])
        plt.show()
    
        
    
        TP.rio.to_raster("D:/OneDrive - Mendelova univerzita v Brně/"
                   "Coherence_VI_Krtiny/clusters/TP/"+
                          str(Area)+".tif")
        
        TN.rio.to_raster("D:/OneDrive - Mendelova univerzita v Brně/"
                   "Coherence_VI_Krtiny/clusters/TN/"+
                          str(Area)+".tif")
