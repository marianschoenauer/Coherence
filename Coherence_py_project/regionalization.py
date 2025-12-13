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
"""
def read_rast(path):
    '''reads in netCDF rastera and reprojects it to reference'''
    return rio.open_rasterio(ROOT + path)\
        .squeeze()\
            .rio.reproject_match(PlanetScope) # by default the nearest

dom_height = read_rast('forest/dom_height/dom_height.tif').rename("dom")

Elev = read_rast('terrain/DGM5.tif').rename("elevation")
Slope = read_rast('terrain/Slope15percent.tif').rename("slope")
Aspect = read_rast('terrain/Aspect15.tif').rename("aspect")
ProfCurv = read_rast('terrain/ProfileCurvature15.tif').rename("curv_pro")
TangCurv = read_rast('terrain/TangCurvature15.tif').rename("curv_tan")

rasters = xr.merge([Elev, PlanetScope, dom_height,
                    Slope, Aspect, ProfCurv, TangCurv])
del Elev, dom_height, Slope, Aspect, ProfCurv, TangCurv
# %% vector data

# --- Load forest stands vector ---
vect_stands_LT = gpd.read_file(ROOT + "forest/stands/"
                               "forest_stands_25833_LT.shp").to_crs(CRS)

# --- Soil data ---
vect_soil = gpd.read_file(ROOT + 'soil_map/commondata/marian/pudni_typy.shp')\
    .to_crs(CRS)\
        .loc[:,['TYP', 'SUBTYP', 'geometry']]

# %%% rasterize

rst_yx = PlanetScope.to_dataframe().reset_index().loc[:,['y','x']]
rst_yx = gpd.GeoDataFrame(rst_yx,
                          geometry = gpd.points_from_xy(x = rst_yx['x'],
                                                        y = rst_yx['y']),
                          crs= CRS)

for vector_layer in [vect_stands_LT, vect_soil]:

    for feature in vector_layer.columns.drop('geometry'):

        sp_dissolved = vector_layer.dissolve(by = feature)
        rst_yx[feature] = None

        for i in range(0,len(sp_dissolved)):
            print(i)
            s_i = sp_dissolved.iloc[[i],:]
            index_i = rst_yx.sjoin(s_i).index
            rst_yx.loc[index_i,feature] = s_i.index.values[0]

soils = xr.Dataset.from_dataframe(rst_yx.set_index(['x', 'y'])\
                                  .drop(columns = 'geometry'))\
    .rio.write_crs(CRS)

# merge to stack

stack = xr.merge([rasters, soils])

pd.Series(stack['LT'].values.flatten()).value_counts()
stack.to_netcdf('D:/stack_rasters_soil.nc')
"""
# %% get stack

"""
st = xr.load_dataset('D:/stack_rasters_soil.nc')\
    .rio.write_crs(CRS)

for var in ['LT', 'TYP', 'SUBTYP']:
    values = pd.factorize(st[var].values.ravel())[0]\
        .reshape(st[var].shape)

    st[var] = (['x','y'], values)
"""

# %% clustering
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
larger_gaps = all_gaps[all_gaps.area > 100].copy()
small_gaps = all_gaps = all_gaps[all_gaps.area < 100].copy()

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
    
    
"""
    clip = st['PlanSco'].rio.clip(AOI['geometry'])
    
    clip.rio.to_raster("D:/OneDrive - Mendelova univerzita v Brně/"
               "Coherence_VI_Krtiny/clusters/tp_plus_buffer/"+
                      str(Area)+".tif")
    
   

    df = clip.to_dataframe()\
        .drop(columns = ['band', 'spatial_ref'])\
            .dropna()

    # Compute Gower distance
    gower_dist = gower.gower_matrix(df)

    # PAM Clustering (KMedoids)
    pam_fit = KMedoids(n_clusters=10, metric='precomputed', random_state=42)
    pam_fit.fit(gower_dist)
    df['cluster'] = pam_fit.labels_

    # print clusters as netCDF
    # reproject_match to PlanetScope, to coerce same coordinates for every Area
    cluster = xr.Dataset.from_dataframe(df[['cluster']])\
        .transpose('y', 'x')\
                .rio.write_crs(CRS)\
                    .rio.reproject_match(PlanetScope)

    # crop it to extent of the AOI
    minx, miny, maxx, maxy = AOI.bounds.to_numpy()[0]
    cluster = cluster.sel(x = slice(minx-20,maxx+20),
                          y = slice(maxy+20, miny-20)).copy()
    
    cluster = cluster.rio.write_crs(PlanetScope.spatial_ref.attrs["crs_wkt"])

    

    cluster.rio.to_raster("D:/OneDrive - Mendelova univerzita v Brně/"
               "Coherence_VI_Krtiny/clusters/tp_plus_buffer_300m/"+
                      str(Area)+".tif")

    del AOI, clip, df, gower_dist, pam_fit, cluster
"""
