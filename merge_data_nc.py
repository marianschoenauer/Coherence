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

windthrows = gpd.read_file("D:/OneDrive - Mendelova univerzita v Brně/"
                           "Coherence_VI_Krtiny/"
                           "shapefiles/gaps.gpkg", layer = "gaps_selected")
windthrows['geometry'] = windthrows['geometry'].buffer(200)


# Sentinel-1
s1_full =  xr.load_dataset(ROOT + "coherence_backscatter_data/netCDF/"
                           "backscatter_stack_Krtiny_10m_2024_dB_32633.nc", 
                           engine = 'h5netcdf')\
    .rename({'gamma0_VV_dB':'VV', 'gamma0_VH_dB':'VH'})\
        .drop_vars(['gamma0_VV','gamma0_VH'])
CRS = s1_full.spatial_ref.attrs['crs_wkt']
#s1_full = s1_full.assign_coords(x = (s1_full.x.values - 30))
        
s1_full = s1_full\
    .rio.write_crs(CRS)
np.diff(s1_full.x)

xmin, ymin, xmax, ymax = windthrows.to_crs(CRS).total_bounds

s1_full = s1_full.sel({'x':slice(xmin, xmax), 'y':slice(ymax,ymin)})

# Sentinel-2
s2_full =  xr.load_dataset(ROOT + "s2/"
                "0.nc", engine = 'h5netcdf')
np.diff(s2_full.x)
s2_full = s2_full\
    .rio.write_crs(s2_full.spatial_ref.attrs['crs_wkt'])\
        .rio.reproject_match(s1_full)
        
s2_full["NDVI"]  = (s2_full["B8"] - s2_full["B4"]) / (s2_full["B8"] + s2_full["B4"])

# Coherence

coherence_full =xr.load_dataset(ROOT + "coherence_backscatter_data/netCDF/"
                           "coherence_stack_Krtiny_10m_2024_32633.nc", 
                           engine = 'h5netcdf')

#coherence_full = coherence_full.assign_coords(x = (coherence_full.x.values - 30))
  

coherence_full = coherence_full\
    .rio.write_crs(coherence_full.spatial_ref.attrs['spatial_ref'])\
        .rio.reproject_match(s1_full)


def coherence_baseline_days(days, Dataset):
    NAME = "coh_" + str(days)
    coh_X = Dataset.sel(pair=Dataset.baseline_days == days)\
        .rename({'coherence':NAME})
    np.diff(coh_X.x)
    coh_X = coh_X.assign_coords(pair = coh_X.slave_time.values)
    coh_X = coh_X.rename({'pair':'time'})
    coh_X = coh_X.drop_vars(['baseline_days', 'pair_name', 'master_time', 'slave_time'])
    return coh_X

coh_12 = coherence_baseline_days(12, coherence_full)
coh_24 = coherence_baseline_days(24, coherence_full)
coh_36 = coherence_baseline_days(36, coherence_full)


conc = xr.merge([s1_full, 
                  s2_full,
                  coh_12,coh_24,coh_36
                  ]).sortby(['x','y','time'])


conc['Rc'] = conc['VH'] - conc['VV']

conc = conc.drop_vars('spatial_ref')
conc = conc.rio.write_crs(s1_full.spatial_ref.attrs['crs_wkt'])\
    .rio.reproject(s1_full.spatial_ref.attrs['crs_wkt'])

conc = conc.assign_coords(time = pd.to_timedelta(conc['time'] - np.datetime64("2024-06-21")))

# %%

month_before = pd.to_timedelta(-4, unit = 'W')
day_0 = pd.to_timedelta(0, unit = 'd')
month_after = pd.to_timedelta(4, unit = 'W')

Mean = conc.sel({"time":slice(month_before, day_0)}).mean(dim = 'time')

Diffs = conc.sel({'time':slice(month_before, month_after)}) - Mean

Diffs["WI"] = Diffs['VV'] + Diffs['VH']

Diffs["WI"].sum(dim = 'time').plot()
plt.show()
# %% assign TN and TP
df_list = []

for PATH in glob.glob(ROOT + "clusters/TN/*.tif"):
    
    AOI_name = PATH.replace(ROOT + "clusters/TN\\","")
    print(AOI_name)
    
    TN = rio.open_rasterio(PATH)
    CRS = TN.spatial_ref.attrs['crs_wkt']
    
     
    TP = rio.open_rasterio(PATH.replace("TN","TP"))\
        .rio.reproject_match(TN)
        
    tp = TP.where(TP != 255, np.nan)\
        .to_dataframe(name = 'PlanSco')\
                    .dropna()\
                        .droplevel('band')
     
    tn = TN.where(TN != 255, np.nan)\
        .to_dataframe(name = 'PlanSco')\
                    .dropna()\
                        .droplevel('band')

            
    sat = Diffs.rio.reproject_match(TN)\
        .to_dataframe()\
            .reset_index('time')\
                .swaplevel()\
                    .sort_index()
            
    tp_sat = sat.loc[tp.index,:].assign(Gap = True, AOI = AOI_name)
    tn_sat = sat.loc[tn.index,:].assign(Gap = False, AOI = AOI_name)
    
    if np.any(tp_sat.index.isin(tn_sat.index)):
        print("pixels overlap")
    else:
        df_list.append(tp_sat)
        df_list.append(tn_sat)


df_nc = pd.concat(df_list)\
    .set_index(['Gap','time'], append = True)\
        .sort_index()

del df_list


# %% set index
df = df_nc.loc[~df_nc.index.duplicated()].copy()
df = df.reset_index()\
    .set_index(['Gap', 'AOI','x', 'y', 'time'])\
        .sort_index()\
            .drop(columns = 'spatial_ref')\
                .unstack('time')
df.columns.levels[0]
# %% plot time series

vars_selected = ['coh_12','coh_24','coh_36',  
                 'VV', 'VH', 'WI',
                 'NDVI', 'B4', 'B8']
fig, ax = plt.subplot_mosaic([['VV', 'coh_12', 'B4'],
                              ['VH', 'coh_24', 'B8'],
                              ['WI','coh_36', 'NDVI']], figsize= (8,10), constrained_layout = True)

for VAR in vars_selected:
    
    df_VAR = df.loc[:,VAR].stack().to_frame(name = VAR).reset_index()

    
    sns.boxplot(df_VAR, x = 'time', y = VAR, hue = 'Gap', showfliers = False, ax = ax[VAR])


    ax[VAR].tick_params(axis='x', rotation=90)

    
    ax['VV'].set_title('Backscatter', fontweight= 'bold')
    ax['coh_12'].set_title('Coherence', fontweight= 'bold')
    ax['B4'].set_title('Sentinel-2', fontweight= 'bold')
    
    if np.isin(VAR, ['VV','VH','B4', 'B8']):
        
        ax[VAR].set_xticklabels([])
        ax[VAR].set_xlabel("")
        
    if VAR != 'B8':
        ax[VAR].legend().remove()
plt.show()

# %% setting

cols_s2 = ['NDVI'] #'B3','B4','B5','B8A','B11','B12',
cols_s1 = ['WI', 'VH','VV'] #'Rc','RVI',
cols_coherence = ['coh_12', 'coh_24', 'coh_36'] #,

sce_data = {'s1':cols_s1,
            'coherence':cols_coherence,
           's1+coherence':cols_s1+cols_coherence,
           's2':cols_s2,
           #'s1+s2': cols_s1 + cols_s2,
           #'s1+s2+coherence': cols_s1+cols_s2+cols_coherence
           }

sce = pd.DataFrame({'cols': sce_data.values()}, index=sce_data.keys())

# %% models

from sklearn.linear_model import RidgeClassifier as model
from sklearn.metrics import f1_score, accuracy_score, fbeta_score
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

results = []
output_settings = []

for Setting in sce.index:
    Setting
    output_aois = []

    for AOI in df.index.levels[1]:
        
        cols = sce.loc[Setting,'cols']
        
        Partition = df.reset_index()['AOI'].isin([AOI]).values
        Partition.sum()

        test = df.loc[Partition,ix[cols, day_0:]].dropna(axis = 1, thresh = 0.8).dropna().copy()
        train = df.loc[~Partition,ix[cols, day_0:]].dropna(axis = 1, thresh = 0.8).dropna().copy()
        
        if np.any(test.reset_index(['Gap','AOI']).index.isin(train.reset_index(['Gap','AOI']).index)):
            print("OVERLAPPING PIXELS")
        
        y = train.reset_index().loc[:,'Gap'].values.flatten()
        X = train.loc[:,cols]

        pos_weight = df.groupby('Gap').size()[False] / df.groupby('Gap').size()[True] * 0.75
        
        weights = X.shape[0] / np.sqrt(X.groupby('Gap').transform("size").values)
                
        pipe = Pipeline([('scaler',StandardScaler()),
                         ('poly', PolynomialFeatures()),
                         ('model',model(random_state = 1))]) #, scale_pos_weight = pos_weight

        model_fit = pipe.fit(X = X.to_numpy(), y = y, model__sample_weight = weights)
        test['pred_' + Setting] = model_fit.predict(test.loc[:,cols])

        output_aois.append(test)
                
        F1 = f1_score(y_true = test.reset_index()['Gap'], y_pred = test['pred_' + Setting], zero_division = 0)
        F05 = fbeta_score(y_true = test.reset_index()['Gap'], y_pred = test['pred_' + Setting], beta = 0.5, zero_division = 0)
        ACC = accuracy_score(y_true = test.reset_index()['Gap'], y_pred = test['pred_' + Setting])
        
        results.append({'setting':Setting,'test_area':AOI,'F1':F1, 'F05': F05, 'ACC':ACC})

        
    output_settings.append(pd.concat(output_aois))

     
output = pd.concat(output_settings, axis = 1).sort_index()

Res = pd.DataFrame(results)
sns.boxplot(Res, x = 'setting', y = 'F1')
plt.show()

from scipy.stats import ttest_rel

Res.set_index('setting', inplace = True)
Res.loc['s1+coherence','F1'].values
pval = str(ttest_rel(Res.loc['s1+coherence','F1'].values,
          Res.loc['s1','F1'].values).pvalue)[:5]


print(Res.groupby('setting')['F1'].describe())
print('Diff S1 vs. S1+coh.:', pval)
print('F1 mean:', Res['F05'].mean())

# %%
"""
for AOI in output.index.levels[1]:
    out = output.loc[ix[:,AOI],:].copy().sort_index()
    
    out.reset_index(['Gap','AOI'])

    out = out.loc[:,out.droplevel(1, axis = 1).columns.str.startswith('pred')].droplevel(1,axis = 1).reset_index(['Gap','AOI']).drop(columns = 'AOI')

    out = out.astype(np.bool_).astype(int)
    
    out = out.swaplevel().sort_index(level = ['y','x'])
            
    ds = xr.Dataset.from_dataframe(out)\
        .sortby(['y','x'])\
            .rio.write_crs(CRS)\
                .rio.reproject(4326)
    
    fig, ax = plt.subplot_mosaic([['Gap','S2'],
                                  ['S1', 'coh_S1']], figsize = (12,10))
                
    ds['Gap'].plot(ax = ax['Gap'])
    ds['pred_s2'].plot(ax = ax['S2'])
    ds['pred_s1'].plot(ax = ax['S1'])
    ds['pred_s1+coherence'].plot(ax = ax['coh_S1'])
    
    ax['Gap'].set_title('Gap ground-truth')
    ax['S2'].set_title('Pred.: NDWI')
    ax['S1'].set_title('Pred.: WI =delta(VV+VH')
    ax['coh_S1'].set_title('Pred.: WI+Coherence')
    plt.show()
    
    
    
    ds.rio.to_raster(ROOT + "preds/" + AOI)
    
                    
    #ds.to_netcdf("D:/" + AOI, engine= 'h5netcdf')
 """   