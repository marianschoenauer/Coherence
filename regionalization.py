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

GAPS = ("D:/OneDrive - Mendelova univerzita v Brně/"
                           "Coherence_VI_Krtiny/"
                           "shapefiles/gaps.gpkg")

CRS = 3035
'''
all_gaps = pd.concat([
    gpd.read_file(GAPS, layer = "canopy_diffs")\
        .assign(Country = 'SLP').to_crs(CRS),
    gpd.read_file(GAPS, layer = "Italy_windthrows")\
        .assign(Country = 'ITA').to_crs(CRS),
    gpd.read_file(GAPS, layer = "Germany_Friederike")\
        .assign(Country = 'GER').to_crs(CRS), 
    ], ignore_index=True).loc[:,['Country', 'geometry']]\
    .to_file(GAPS, layer = 'all_gaps')
'''    
all_gaps = gpd.read_file(GAPS, layer = 'all_gaps').to_crs(CRS)

## exclude small areas
MMU = 20**2
larger_gaps = all_gaps[all_gaps.area > MMU].copy()
small_gaps = all_gaps[all_gaps.area < MMU].copy()

## make one geometry
LARGER_GAPS = larger_gaps.union_all()
SMALL_GAPS = small_gaps.union_all()

AOIs = gpd.read_file(GAPS, layer = "AOIs_individual_gaps_3035")
AOIs['AOI'] = AOIs['Country'] + "_" + AOIs['ID_No'].astype(str)
AOIs = AOIs.set_index('AOI')

gaps_conifer = gpd.read_file(GAPS, layer = "gap_in_conifers").to_crs(CRS)


# gap sizes in AOIs

sizes = larger_gaps.sjoin(AOIs.loc[:,['ID_No', 'geometry']])
sizes['Area_ha'] = sizes.area / 10000

sizes.groupby(['Country', 'ID_No'])['Area_ha'].sum().groupby('Country').describe()


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



for AOI, row, in AOIs.iterrows():#'USA',
    print(AOI)
    A = row['Country']

    fm = xr.load_dataset(ROOT + 'DynWorld/cropped/'+AOI+'.nc', engine = 'h5netcdf')
    CRS_DS = fm['spatial_ref'].attrs['crs_wkt']
    fm = fm.rio.write_crs(CRS_DS).rio.reproject(CRS)
        
    forestMask = fm.where(fm.label == 1)['label'].squeeze()

    forestMask = forestMask.rio.clip(gaps_conifer, invert = True)
    
    #  select TN and TP

    TP = forestMask.rio.clip([LARGER_GAPS])
    TN = forestMask.rio.clip([LARGER_GAPS], invert = True)  
    TN = TN.rio.clip([SMALL_GAPS], invert = True)
    '''
    fig, ax = plt.subplots(2,1)
    
    TP.plot(ax = ax[0])
    TN.plot(ax = ax[1])
    ax[0].set_title(AOI)
    plt.show()       
   '''
    TP.rio.to_raster("D:/OneDrive - Mendelova univerzita v Brně/"
               "Coherence_VI_Krtiny/clusters/TP/"+
                      str(AOI)+".tif")
    
    TN.rio.to_raster("D:/OneDrive - Mendelova univerzita v Brně/"
               "Coherence_VI_Krtiny/clusters/TN/"+
                      str(AOI)+".tif")
