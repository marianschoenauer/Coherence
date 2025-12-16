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
CRS =25833

ROOT = ("D:/OneDrive - Mendelova univerzita v Brně/"
"IGA Team - UFE Wind Throw - General/GIS/")


# %% raster data
PlanetScope = rio.open_rasterio(ROOT + "forest/canopy/"
                                "TreeSpecies_PlanetScope.tif")\
    .squeeze()\
        .rio.reproject(CRS)\
            .rename("PlanSco")
## replace 255 by np.nan. when where is applied to xarray, the condition is 
## inverse
PlanetScope = PlanetScope.where(PlanetScope != 255., np.nan)
PlanetScope.attrs["_FillValue"] = np.nan
PlanetScope.plot()


# %% select TN and TP
import os, glob
for path in glob.glob("D:/OneDrive - Mendelova univerzita v Brně/"
                      "Coherence_VI_Krtiny/clusters/TN/*"):
    os.remove(path)
for path in glob.glob("D:/OneDrive - Mendelova univerzita v Brně/"
                      "Coherence_VI_Krtiny/clusters/TP/*"):
    os.remove(path)


AOIs = gpd.read_file("D:/OneDrive - Mendelova univerzita v Brně/"
                           "Coherence_VI_Krtiny/"
                           "shapefiles/gaps.gpkg", layer = "AOIs")\
    .to_crs(CRS)\
        .set_index('AOI')

all_gaps = gpd.read_file("D:/OneDrive - Mendelova univerzita v Brně/"
                           "Coherence_VI_Krtiny/"
                           "shapefiles/gaps.gpkg", layer = "canopy_diffs")\
    .to_crs(CRS)
## exclude small areas
one_pixel_coherence = 40*40

larger_gaps = all_gaps[all_gaps.area > one_pixel_coherence*2].copy()
small_gaps = all_gaps = all_gaps[all_gaps.area < one_pixel_coherence*2].copy()

## make one geometry

LARGER_GAPS = larger_gaps.union_all()
SMALL_GAPS = small_gaps.union_all()

# Loop through AOIs
for Area, row in AOIs.iterrows():

    print(f'Processing area {Area}...')
      
    ds_AOI = PlanetScope.rio.clip(row)

    TP = ds_AOI.rio.clip([LARGER_GAPS])
    TN = ds_AOI.rio.clip([LARGER_GAPS], invert = True)
    TN = TN.rio.clip([SMALL_GAPS], invert = True)

    TP.rio.to_raster("D:/OneDrive - Mendelova univerzita v Brně/"
               "Coherence_VI_Krtiny/clusters/TP/"+
                      str(Area)+".tif")
    
    TN.rio.to_raster("D:/OneDrive - Mendelova univerzita v Brně/"
               "Coherence_VI_Krtiny/clusters/TN/"+
                      str(Area)+".tif")
