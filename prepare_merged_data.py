# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:49:22 2026

@author: Marian Schonauer
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import rioxarray as rio
import pandas as pd
import seaborn as sns
from skimage import feature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, precision_score, \
    f1_score, fbeta_score, recall_score
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier  as model
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC as model
#from xgboost import XGBClassifier
#from sklearn.preprocessing import PolynomialFeatures
ix = pd.IndexSlice

USER = "Marian"

if USER == "Marian":
    ROOT = "D:/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/"
    NC = ROOT + 'satellite_data/merges_cropped/'
else:
    ROOT = "C:/Users/Lika/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/"
    NC = "C:/Users/Lika/Desktop/py_coherence/NCs/"

FIGS, TABS = ROOT + "Manuscript/figures/", ROOT + "Manuscript/tables/"

AOIs = gpd.read_file(ROOT + "shapefiles/gaps.gpkg", layer = "AOIs_3857")\
    .set_index('AOI')

# %% create Mean diffs

Df_nc = []

for AOI in AOIs.index:
    A = AOI[:3]
    
    conc = xr.load_dataset(NC + AOI + '.nc', engine = "h5netcdf")
    conc = conc.rio.write_crs(conc.spatial_ref.attrs['crs_wkt'])
    conc = conc.assign(WI = conc['VV'] + conc['VH'])
    conc = conc.assign(RRVI = conc['VH']/conc['VV'])
    conc = conc.assign(NDVI = (conc["B8"] - conc["B4"]) / (conc["B8"] + conc["B4"]))
    
    month_before = -pd.to_timedelta(4, unit = 'W')
    day_0 = pd.to_timedelta(0, unit = 'd')
    delay  = pd.to_timedelta(7, unit = 'd')
    month_after = pd.to_timedelta(4, unit = 'W')
    
    Before = conc.sel({"time":slice(month_before, day_0)}).mean(dim = 'time')
    After = conc.sel({"time":slice(delay, month_after)}).mean(dim = 'time')
    
    Mean = Before - After
    ds = Mean.fillna(Mean.mean())
    
    del month_before, month_after, delay, Before, After, Mean, conc
    
# Segmentation features
    
    list_da = []
    coords = ds.coords
    
    for feat_name in list(ds.data_vars):
        feat = feature.multiscale_basic_features(ds[feat_name].values,
                                          intensity=True,
                                          edges=False,
                                          texture=True,
                                          sigma_min=1,
                                          sigma_max=16,
                                          num_sigma = 5)
    
        np_name = np.array(feat_name)
        np_sigmas = np.array(["1","2","3","4","5"])
    
        feat_names = np.concatenate((
            np_name + np.array(["_intensity_"])+ np_sigmas,
            np_name + np.array(["_texture1_"])+ np_sigmas,
            np_name + np.array(["_texture2_"])+ np_sigmas))
    
        coords_i = coords.assign(feature = feat_names)
    
        da_feat = xr.DataArray(feat, coords_i[['y','x','feature']].coords)
        da_feat.name = feat_name
    
        list_da.append(da_feat)
    
    ds = xr.merge(list_da)
    
# labelling
    df_list = []
    

    TN = rio.open_rasterio(ROOT + "clusters/TN/"+AOI+".tif")

    TP = rio.open_rasterio(ROOT + "clusters/TP/"+AOI+".tif")\
        .rio.reproject_match(TN)

    tp = TP.where(TP != 255, np.nan)\
        .to_dataframe(name = 'PlanSco')\
                    .dropna()\
                        .droplevel('band')

    tn = TN.where(TN != 255, np.nan)\
        .to_dataframe(name = 'PlanSco')\
                    .dropna()\
                        .droplevel('band')

    sat = ds.transpose('feature', 'y', 'x')\
        .rio.reproject_match(TN)\
            .to_dataframe()\
                .drop(columns = 'spatial_ref')\
                    .unstack(0)\
                        .swaplevel()\
                            .sort_index()

    tp_sat = sat.loc[tp.index,:].assign(Gap = True, Country = A, AOI = AOI)
    tn_sat = sat.loc[tn.index,:].assign(Gap = False, Country = A, AOI = AOI)

    if np.any(tp_sat.index.isin(tn_sat.index)):
        print("pixels overlap")
    else:
        df_list.append(tp_sat)
        df_list.append(tn_sat)
    
    df_nc = pd.concat(df_list)\
        .set_index(['Gap'], append = True)\
            .sort_index()
            
    Df_nc.append(df_nc)
    
    del df_list, tp_sat, tn_sat, sat, TN, TP, tp, tn

# %% set index

Df_nc_c = pd.concat(Df_nc)

df = Df_nc_c.loc[~Df_nc_c.index.duplicated()].copy()
df = df.reset_index()\
    .set_index(['Country', 'Gap', 'AOI', 'x', 'y'])\
        .sort_index()\
            .dropna(how = 'all', axis = 1)

df = df.fillna(df.mean())

df.to_pickle(ROOT + 'df.pkl')
