# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:25:27 2025

@author: Marian Schonauer
"""

import glob
import geopandas as gpd
import xarray as xr
import rioxarray as rio
import pandas as pd
ix = pd.IndexSlice
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
CRS =5514

ROOT = ("D:/OneDrive - Mendelova univerzita v Brně/"
"IGA Team - UFE Wind Throw - General/GIS/")


# %% test

ds = xr.load_dataset("D:/Coherence_VI_Krtiny/coherence_backscatter_data/"
                "netCDF/coherence_stack_Krtiny_10m_pairs.nc")
crs_wkt = ds.spatial_ref.attrs['crs_wkt']
ds = ds.rio.write_crs(crs_wkt)



for i in ds.pair.values:

    ds_i = ds.isel(pair=i)
    t1 = ds_i['master_time'].values
    t2 = ds_i['slave_time'].values
    
    times = np.sort(([t1,t2]))
    
    print(t2 > t1)
    
    
# %%% read tiles

df_list = []

for PATH in glob.glob("D:/Coherence_VI_Krtiny/clusters/tp_plus_buffer_300m/*.nc"):

    AOI = xr.load_dataset(PATH)
    AOI = AOI.rio.write_crs(AOI.spatial_ref.attrs['crs_wkt'])

    tp = xr.load_dataset(PATH.replace("tp_plus_buffer_300m","tp"))
    tp = tp.rio.write_crs(tp.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(AOI)\
                .to_dataframe()\
                    .dropna()

    tn = xr.load_dataset(PATH.replace("tp_plus_buffer_300m","tn_plus_buffer_300m"))
    tn = tn.rio.write_crs(tn.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(AOI)\
                .to_dataframe()\
                    .dropna()
                    
    ds_i = ds.rio.reproject_match(AOI)
    ds_i_df = ds_i.to_dataframe()

     

    tp_sat = ds_i_df.loc[ds_i_df.index.droplevel('pair').isin(tp.index),:].assign(Gap = True, 
                  ID = PATH.replace("D:/Coherence_VI_Krtiny/clusters/tp_plus_buffer_300m\\",""))
    tn_sat = ds_i_df.loc[ds_i_df.index.droplevel('pair').isin(tn.index),:].assign(Gap = False, 
                  ID = PATH.replace("D:/Coherence_VI_Krtiny/clusters/tp_plus_buffer_300m\\",""))

    df_list.append(tp_sat)
    df_list.append(tn_sat)

    del tp, tn, tp_sat, tn_sat

df = pd.concat(df_list)

# %% set index
df = df.reset_index()\
    .set_index(['ID', 'pair', 'Gap', 'x', 'y'])\
        .sort_index()
        
res = []
    
for ID in df.index.levels[0]:
           
    for pair in df.index.levels[1]:
        df_pair = df.loc[ix[ID,pair],:]
        
        gap = df_pair.loc[True,'coherence'].values
        nogap = df_pair.loc[False,'coherence'].values
        
        df_pair['baseline_days'][0]
        
        df_pair.columns
        
        res.append({
        'ID':ID,
        'pair':pair,
         'baseline_days':df_pair['baseline_days'][0],
         'master_time':df_pair['master_time'][0],
         'slave_time':df_pair['slave_time'][0],
         'diff':gap.mean() - nogap.mean()})
    
  
    
res = pd.DataFrame(res).set_index(['ID','pair'])


# %%%
res['baseline_days'] = res['baseline_days'].astype(str)

res = res.loc[(res.slave_time > np.datetime64("2024-03-01")) & (res.slave_time < np.datetime64("2024-10-01"))]
  
sns.scatterplot(res, x = 'slave_time', y = 'diff', hue = 'baseline_days')
plt.xticks(rotation=90)
plt.axhline(0)
# %% raster data

df_list = []

for PATH in glob.glob("D:/Coherence_VI_Krtiny/clusters/tp_plus_buffer_300m/*.nc"):
    
    AOI = xr.load_dataset(PATH)
    AOI = AOI.rio.write_crs(AOI.spatial_ref.attrs['crs_wkt'])


    coherence =xr.load_dataset("D:/Coherence_VI_Krtiny/coherence_backscatter_data/"
                    "netCDF/coherence_stack_Krtiny_10m_pairs.nc")
    coherence = coherence\
        .rio.write_crs(coherence.spatial_ref.attrs['spatial_ref'])\
            .rio.reproject_match(AOI)\
                .groupby(coherence.slave_time.dt.strftime('%Y%m%d')).mean()                
                                 
    s1 =  xr.load_dataset("D:/Coherence_VI_Krtiny/s1/"
                    "0.nc")
    s1 = s1.rio.write_crs(s1.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(AOI)\
                .groupby(s1.time.dt.strftime('%Y%m%d')).mean()


    s2 =  xr.load_dataset("D:/Coherence_VI_Krtiny/s2/"
                    "0.nc")
    s2 = s2.rio.write_crs(s2.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(AOI)\
                .groupby(s2.time.dt.strftime('%Y%m%d')).mean()
    
    df_sub = xr.merge([coherence, s1, s2])\
        .to_dataframe()\
            .astype(float)\
                    .sort_index()
                    
    df_list.append(df_sub)
    del df_sub,  s2

df_nc = pd.concat(df_list)\
        .drop(columns = 'spatial_ref')\
            .swaplevel()\
                .sort_index()

df_nc = df_nc.loc[~df_nc.index.duplicated()]

del df_list

df_nc = df_nc.reset_index('strftime')
df_nc['time'] = pd.to_datetime(df_nc['strftime']).values

df_nc = df_nc.set_index('time', append = True).drop(columns = 'strftime')


# %% indices

df_nc.loc[:,'NDVI'] = (df_nc.B8 - df_nc.B4) / (df_nc.B8 + df_nc.B4)
df_nc.loc[:,'NDWI'] = (df_nc.B3 - df_nc.B8A) / (df_nc.B3 + df_nc.B8A)
#df_nc.loc[:,'Rc'] = df_nc.VH / df_nc.VV
##df_nc.loc[:,'RVI'] = (4 * df_nc.VH) / (df_nc.VV + df_nc.VH)
#df_nc.loc[:,'RNDVI'] = (df_nc.VH - df_nc.VV) / (df_nc.VH + df_nc.VV)       

# %% read tiles

df_list = []

for PATH in glob.glob("D:/Coherence_VI_Krtiny/clusters/tp_plus_buffer_300m/*.nc"):

    AOI = xr.load_dataset(PATH)
    AOI = AOI.rio.write_crs(AOI.spatial_ref.attrs['crs_wkt'])

    tp = xr.load_dataset(PATH.replace("tp_plus_buffer_300m","tp"))
    tp = tp.rio.write_crs(tp.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(AOI)\
                .to_dataframe()\
                    .dropna()

    tn = xr.load_dataset(PATH.replace("tp_plus_buffer_300m","tn_plus_buffer_300m"))
    tn = tn.rio.write_crs(tn.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(AOI)\
                .to_dataframe()\
                    .dropna()

    tp_sat = df_nc.loc[df_nc.index.droplevel('time').isin(tp.index),:].assign(Gap = True, 
                  ID = PATH.replace("D:/Coherence_VI_Krtiny/clusters/tp_plus_buffer_300m\\",""))
    tn_sat = df_nc.loc[df_nc.index.droplevel('time').isin(tn.index),:].assign(Gap = False, 
                  ID = PATH.replace("D:/Coherence_VI_Krtiny/clusters/tp_plus_buffer_300m\\",""))

    df_list.append(tp_sat)
    df_list.append(tn_sat)

    del tp, tn, tp_sat, tn_sat

df = pd.concat(df_list)

# %% set index
df = df.reset_index()\
    .set_index(['ID','Gap', 'x', 'y', 'time'])\
        .sort_index()

# %% plot time series
fig, ax = plt.subplot_mosaic([['coherence'],['RNDVI'], ['NDVI']])

for var in ['NDVI']:
    sns.lineplot(df.reset_index(), x = 'time', y = var, hue = 'Gap', ax = ax[var])
    
    ax[var].tick_params(axis='x', rotation=90)
    ax[var].axvline(np.datetime64("2024-06-21"))

# %% diffs
df_var = df.loc[:,['NDVI']].dropna().sort_index(level = 'time')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_var.loc[:,['NDVI']] = scaler.fit_transform(df_var)

coords = df_var.reset_index()[['x','y']]\
    .drop_duplicates()

df['diff1week'] = np.nan
df['diff1month'] = np.nan

preds = []

for xy in coords.iterrows():

    dfs = df_var.loc[ix[:,:,xy[1]['x'], xy[1]['y']], ['NDVI']].sort_index()

    theHappening = np.datetime64("2024-06-21 18:00")
    month_before = np.datetime64("2024-05-21 00:00")
    week_after = np.datetime64("2024-06-28")
    month_after = np.datetime64("2024-07-21")

    before = dfs.loc[ix[:,:,:,:,month_before:theHappening],:]
    week_after = dfs.loc[ix[:,:,:,:,theHappening:week_after],:]
    month_after = dfs.loc[ix[:,:,:,:,theHappening:month_after],:]

    if (before.shape[0] > 0) & (week_after.shape[0] > 0):

        diff_week = week_after.mean() - before.mean()
        diff_month = month_after.mean() - before.mean()

        preds.append(
        pd.DataFrame({'week_after': diff_week['NDVI'],
                      'month_after': diff_month['NDVI']},
                     index = dfs.droplevel('time').index[[0]])
        )

Pred = pd.concat(preds)

sns.boxplot(Pred.reset_index(), x = 'Gap', y = 'week_after')

for ID in Pred.index.levels[0]:
    print(ID)
    expot = Pred.loc[ID,:]\
        .reset_index('Gap')\
            .sort_index()

    expot['Gap'] = np.where(expot['Gap'], 1, 0)


    ds = xr.Dataset\
        .from_dataframe(expot)\
            .sortby(['x','y'])\
                .transpose('y', 'x')\
                    .rio.write_crs(CRS)\
                        .rio.reproject(CRS)

    fig, ax = plt.subplots(1,3, figsize = (8,3))
    ds['Gap'].plot(ax = ax[0])
    ds['month_after'].plot(ax = ax[1])
    ds['month_after'].plot(ax = ax[2])


    ds.to_netcdf("D:/Coherence_VI_Krtiny/preds/" + ID + ".nc")
