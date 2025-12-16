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



ROOT = ("D:/OneDrive - Mendelova univerzita v Brně/"
           "Coherence_VI_Krtiny/")


# %% raster data

df_list = []


time_start, time_end= pd.Timestamp("2024-04-21"),  pd.Timestamp("2024-08-21")

coherence_full =xr.load_dataset(ROOT + "coherence_backscatter_data/netCDF/"
                           "coherence_stack_Krtiny_10m_2024_32633.nc", 
                           engine = 'h5netcdf')
coherence_full = coherence_full\
    .rio.write_crs(coherence_full.spatial_ref.attrs['spatial_ref'])

coherence_full['slave_time'].values = pd.to_datetime(coherence_full['slave_time'].dt.strftime('%Y%m%d'))
coherence_full = coherence_full.assign_coords(pair = coherence_full.slave_time.values)
coherence_full = coherence_full.rename({'pair':'time'})
coherence_full =  coherence_full\
    .sortby('time').sel(time=slice(time_start, time_end))

"\backscatter_stack_Krtiny_10m_2024_32633.nc"

s1_full =  xr.load_dataset(ROOT + "coherence_backscatter_data/netCDF/"
                           "backscatter_stack_Krtiny_10m_2024_dB_32633.nc", 
                           engine = 'h5netcdf')\
    .rename({'gamma0_VV_dB':'VV', 'gamma0_VH_dB':'VH'})
    

s1_full = s1_full\
    .rio.write_crs(s1_full.spatial_ref.attrs['crs_wkt'])\
        .reset_index(['time'])
s1_full['time'] = pd.to_datetime(s1_full['time'].dt.strftime('%Y%m%d'))
s1_full =  s1_full.sortby('time').sel(time=slice(time_start, time_end))


s2_full =  xr.load_dataset(ROOT + "s2/"
                "0.nc", engine = 'h5netcdf')
s2_full = s2_full\
    .rio.write_crs(s2_full.spatial_ref.attrs['crs_wkt'])\
        .reset_index(['time'])
s2_full['time'] = pd.to_datetime(s2_full['time'].dt.strftime('%Y%m%d'))
s2_full =  s2_full.sortby('time').sel(time=slice(time_start, time_end))


for PATH in glob.glob(ROOT + "clusters/tp_plus_buffer_300m/*.nc"):
    
    AOI_name = PATH.replace(ROOT + "clusters/tp_plus_buffer_300m\\","")
    print(AOI_name)
    
    AOI = xr.load_dataset(PATH, engine = 'h5netcdf')
    
    CRS = AOI.spatial_ref.attrs['crs_wkt']
    AOI = AOI.rio.write_crs(AOI.spatial_ref.attrs['crs_wkt'])
    
    coherence = coherence_full\
            .rio.reproject_match(AOI)\
                .to_dataframe()\
                    .set_index(['baseline_days'], append = True)\
                        .loc[:,'coherence']\
                            .unstack(['baseline_days'])
    coherence.columns = coherence.columns.values.astype(str)
                                 
    s1 = s1_full.rio.reproject_match(AOI)\
        .to_dataframe()\
            .groupby(['time','x','y']).mean()\
                .drop(columns = 'spatial_ref')
            
    s2 = s2_full.rio.reproject_match(AOI)\
        .to_dataframe()\
            .groupby(['time','x','y']).mean()\
                .drop(columns = 'spatial_ref')
                
    df_nc = pd.concat([coherence, s1, s2], axis = 1).assign(AOI = AOI_name)
    
    tp = xr.load_dataset(PATH.replace("tp_plus_buffer_300m","tp"), engine = 'h5netcdf')
    tp = tp.rio.write_crs(tp.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(AOI)\
                .to_dataframe()\
                    .dropna()

    tn = xr.load_dataset(PATH.replace("tp_plus_buffer_300m","tn_plus_buffer_300m"), engine = 'h5netcdf')
    tn = tn.rio.write_crs(tn.spatial_ref.attrs['crs_wkt'])\
            .rio.reproject_match(AOI)\
                .to_dataframe()\
                    .dropna()

    tp_sat = df_nc.loc[df_nc.index.droplevel('time').isin(tp.index),:].assign(Gap = True)
    tn_sat = df_nc.loc[df_nc.index.droplevel('time').isin(tn.index),:].assign(Gap = False)
    
    df_list.append(tp_sat)
    df_list.append(tn_sat)


df_nc = pd.concat(df_list)\
    .set_index('Gap', append = True)\
        .sort_index()
    


df = df_nc.loc[~df_nc.index.duplicated()].copy()

del df_list, df_nc

# %% indices

df.eval("NDVI = (B8 - B4) / (B8 + B4)", inplace = True)
df.eval("NDWI = (B8A - B11) / (B8A + B11)", inplace = True)
df.eval("Rc = VH / VV", inplace = True)
df.eval("RNDVI = (VH - VV) / (VH + VV)", inplace = True)

# %% set index
df = df.reset_index()\
    .set_index(['Gap', 'AOI','x', 'y', 'time'])\
        .sort_index()

# %% plot time series
fig, ax = plt.subplot_mosaic([['RNDVI', '12'], 
                              ['NDVI', '24'],
                              ['B8', '36'],
                              ['VV', '48'],
                              ['VH','60']], figsize= (8,10))

for var in ['NDVI', 'RNDVI', 'B8','VV','VH', '12','24','36','48','60']:

    sns.lineplot(df.reset_index(), x = 'time', y = var, hue = 'Gap', ax = ax[var])
    
    ax[var].tick_params(axis='x', rotation=90)
    ax[var].axvline(pd.Timestamp("2024-06-21"))
    ax[var].set_xlim(time_start, time_end)

# %% differences per pixel
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

Diffs = []

vars_selected = ['12',      '24',    
                 'VV',    'VH', 
                 'NDWI']

month_before = np.datetime64("2024-05-21 00:00")
theHappening = np.datetime64("2024-06-21 00:00")

for VAR in vars_selected:

    Diffs_VAR = []
    df_var = df.loc[:,[VAR]].dropna().sort_index(level = 'time')

    df_var.loc[:,[VAR]] = scaler.fit_transform(df_var)

    coords = df_var.reset_index()[['x','y']]\
        .drop_duplicates()

    print(VAR)

    for xy in coords.iterrows():

        X,Y = xy[1]['x'], xy[1]['y']

        dfs = df_var.loc[ix[:,:,X, Y], :].sort_index()

        before = dfs.loc[ix[:,:,:,:,month_before:theHappening],:]

        for delay_int in np.arange(5):
            try:
                after = dfs.loc[ix[:,:,:,:,theHappening:],:].iloc[[delay_int],:]
                diff = after - before.mean()
                Diffs_VAR.append(diff)

            except:
                pass
            
    Diffs.append(pd.concat(Diffs_VAR))

# %% plot Diffs

Diffs_c = pd.concat(Diffs, axis = 1)

Diffs_c.loc[:,Diffs_c.columns] = scaler.fit_transform(Diffs_c)

Diffs_c[Diffs_c>5] = np.nan
Diffs_c[Diffs_c< -5] = np.nan


Diffs_c['WI'] = Diffs_c['VV'] + Diffs_c['VH']


Diffs_c = Diffs_c.assign(delay = (Diffs_c.reset_index()['time'] - theHappening).values).set_index('delay',append = True).sort_index(level = 'delay')

# %%%

fig, ax = plt.subplot_mosaic(  [['VV', '12', 'NDVI'],
                                ['VH', '24', 'NDWI'],
                                ['WI','36', 'B8' ],
                                ], figsize = (8,7), constrained_layout = True)
vars_selected.append('WI')
for VAR in vars_selected:

    sns.boxplot(Diffs_c.reset_index(), x = 'delay', y = VAR, hue = 'Gap', ax = ax[VAR])
    ax[VAR].tick_params(axis='x', rotation=90)
    
    ax['12'].set_title('Coherence', fontweight= 'bold')
    ax['VV'].set_title('Backscatter', fontweight= 'bold')
    ax['NDVI'].set_title('Sentinel-2', fontweight= 'bold')
    
    if np.isin(VAR, ['VV','VH','12','24','NDVI', 'NDWI']):
        
        ax[VAR].set_xticklabels([])
        ax[VAR].set_xlabel("")
        
    if VAR != 'B8':
        ax[VAR].legend().remove()

# %% setting
df.columns = df.columns.astype(str)

cols_s2 = ['NDWI'] #'B3','B4','B5','B8A','B11','B12',
cols_s1 = ['WI'] #'Rc','RVI',
cols_coherence = ['12'] #,


sce_data = {'s1':cols_s1,
            'coherence':cols_coherence,
           's1+coherence':cols_s1+cols_coherence,
           's2':cols_s2,
           's1+s2': cols_s1 + cols_s2,
           's1+s2+coherence': cols_s1+cols_s2+cols_coherence}


sce = pd.DataFrame({'cols': sce_data.values()}, index=sce_data.keys())

# %% models

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier as model
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


Diffs_c_time = Diffs_c.sort_index().loc[ix[:,:,:,:,:,pd.Timedelta(1,'d'):pd.Timedelta(31,'d')],:].droplevel('time')
results = []
output_settings = []


for Setting in sce.index:
    Setting
    output_aois = []

    for AOI in Diffs_c_time.index.levels[1]:
        
        cols = sce.loc[Setting,'cols']
        
        Partition = Diffs_c_time.reset_index()['AOI'].isin([AOI]).values
        Partition.sum()

        test = Diffs_c_time.loc[Partition,cols].unstack('delay').dropna(axis = 1, thresh = 0.8).dropna().copy()
        train = Diffs_c_time.loc[~Partition,cols].unstack('delay').dropna(axis = 1, thresh = 0.8).dropna().copy()
        
        y = train.reset_index().loc[:,'Gap'].values.flatten()
        X = train.loc[:,cols]
        
        
        sample_weights = 1/ np.sqrt(X.assign(n=1).groupby('Gap').transform(np.size)['n']).values.flatten()
        
        
        pipe = Pipeline([('scaler',StandardScaler()),
                         #('poly', PolynomialFeatures()),
                         ('model',model(random_state = 1))])

        model_fit = pipe.fit(X = X.to_numpy(), y = y, model__sample_weight = sample_weights)
        test['pred_' + Setting] = model_fit.predict(test.loc[:,cols])

        
        output_aois.append(test)
        
   
                
        F1 = f1_score(y_true = test.reset_index()['Gap'], y_pred = test['pred_' + Setting], zero_division = 0)
        ACC = accuracy_score(y_true = test.reset_index()['Gap'], y_pred = test['pred_' + Setting])
        
        results.append({'setting':Setting,'test_area':AOI,'F1':F1, 'ACC':ACC})

        
    output_settings.append(pd.concat(output_aois))

     
output = pd.concat(output_settings, axis = 1).sort_index()

Res = pd.DataFrame(results)
sns.boxplot(Res, x = 'setting', y = 'F1')

Res.groupby('setting')['F1'].describe()

# %%

for AOI in output.index.levels[1]:
    out = output.loc[ix[:,AOI],:].copy()
        
    out.reset_index(['Gap','AOI']).sort_index()

    out = out.loc[:,out.droplevel(1, axis = 1).columns.str.startswith('pred')].droplevel(1,axis = 1).reset_index(['Gap','AOI']).drop(columns = 'AOI')

    out = out.astype(np.bool_).astype(int)
    
    out = out.swaplevel().sort_index(level = ['y','x'])
            
    ds = xr.Dataset.from_dataframe(out)\
        .sortby(['y','x'])\
            .rio.write_crs(CRS)\
                .rio.reproject(4326)
    
    fig, ax = plt.subplots(1,2, figsize = (8,4))
                
    ds['Gap'].plot(ax = ax[0])
    ds['pred_s1+coherence'].plot(ax = ax[1])
    
    plt.show()
                    
    #ds.to_netcdf("D:/" + AOI, engine= 'h5netcdf')
    