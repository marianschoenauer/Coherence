# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 19:06:17 2025

@author: Marian Schonauer
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
#import geopandas as gpd
import xarray as xr
import rioxarray as rio
import pandas as pd
ix = pd.IndexSlice
import seaborn as sns

ROOT = ("D:/OneDrive - Mendelova univerzita v Brně/"
           "Coherence_VI_Krtiny/")

conc = xr.load_dataset(ROOT + "conc.nc", engine = "h5netcdf")


# %%

month_before = pd.to_timedelta(-6, unit = 'W')
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
# %% plot time series

vars_selected = ['coh_12','coh_24','coh_36',
                 'VV', 'VH', 'WI',
                 'NDVI', 'B4', 'B8']

fig, ax = plt.subplot_mosaic([['VV', 'coh_12', 'B4'],
                              ['VH', 'coh_24', 'B8'],
                              ['WI','coh_36', 'NDVI']], figsize= (8,10),
                             constrained_layout = True)

for VAR in vars_selected:

    df_VAR = df.loc[ix[:,'test_BYC.tif'],VAR].stack()\
        .to_frame(name = VAR).reset_index()
    sns.boxplot(df_VAR, x = 'time', y = VAR, hue = 'Gap',
                showfliers = False, ax = ax[VAR])

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

sce_data = {'coherence':cols_coherence,
            's1':cols_s1,
           's1+coherence':cols_s1+cols_coherence,
           #'s2':cols_s2,
           #'s2+s1': cols_s1 + cols_s2,
           #'s2+coherence': cols_s1+cols_s2+cols_coherence
           }

sce = pd.DataFrame({'cols': sce_data.values()}, index=sce_data.keys())

# %% models

from sklearn.linear_model import RidgeClassifier as model
from sklearn.metrics import f1_score, fbeta_score
#from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

tests = []
output_settings = []

for Setting in sce.index:

    output_aois = []

    cols = sce.loc[Setting,'cols']
    print(cols)

    Partition = df.reset_index()['AOI'].isin(['train.tif']).values
    Partition.sum()

    train = df.loc[Partition,ix[cols, day_0:]].dropna(axis = 1, thresh = 0.8)\
        .dropna().copy()
    test = df.loc[~Partition,ix[cols, day_0:]].dropna(axis = 1, thresh = 0.8)\
        .dropna().copy()

    if np.any(test.reset_index(['Gap','AOI']).index.isin(
            train.reset_index(['Gap','AOI']).index)):
        print("OVERLAPPING PIXELS")

    y = train.reset_index().loc[:,'Gap'].values.flatten()
    X = train.loc[:,cols]

    pos_weight = df.groupby('Gap').size()[False] / df.groupby('Gap').size()[True]
    weights = np.where(y, pos_weight, 1)

    pipe = Pipeline([('scaler',StandardScaler()),
                     ('poly', PolynomialFeatures()),
                     ('model',model(random_state = 1, solver = 'auto'))]) 
#, scale_pos_weight = pos_weight

    model_fit = pipe.fit(X = X.to_numpy(), y = y,
                         model__sample_weight = weights)
    test['pred_' + Setting] = model_fit.predict(test.loc[:,cols])

    tests.append(test)

    del train, test, y, X, pos_weight, weights, pipe, model_fit

Test = pd.concat(tests, axis=1)
Test.index = Test.index.remove_unused_levels()

# %% validation metrics

from sklearn.metrics import matthews_corrcoef, precision_score

results_val = []

for Setting in sce.index:
    for AOI in Test.index.levels[1].values:

        y_true = np.array(Test.loc[ix[:,AOI],:].index.get_level_values('Gap'),\
                          dtype = np.bool_)
        y_pred = np.array(Test.loc[ix[:,AOI],"pred_"+Setting].values.flatten(),\
                          dtype = np.bool_)

        F1 = f1_score(y_true = y_true, y_pred = y_pred, zero_division = 0)
        F05 = fbeta_score(y_true = y_true, y_pred = y_pred, beta = 0.5, \
                          zero_division = 0)
        Precision = precision_score(y_true = y_true, y_pred = y_pred, \
                                    zero_division = 0)
        MCC = matthews_corrcoef(y_true = y_true, y_pred = y_pred)

        results_val.append({'setting':Setting,
                        'test_area':AOI,
                        'F1':F1,
                        'F05': F05,
                        'Precision':Precision,
                        'MCC':MCC})

VAL = pd.DataFrame(results_val).set_index(['setting','test_area'])
VAL.columns.name = 'Metric'

sns.boxplot(VAL.stack(level = 'Metric').to_frame(name = "value"), \
            x = 'setting', y = 'value', hue = 'Metric')
plt.show()

from scipy.stats import ttest_rel

print('F05', VAL.groupby('setting')['F05'].describe())
print('F1', VAL.groupby('setting')['F1'].describe())

pval = str(ttest_rel(VAL.loc['s1+coherence','F1'].values,
          VAL.loc['s1','F1'].values).pvalue)[:5]
print('Diff S1 vs. S1+coh.:', pval)
print('F1 mean:', VAL['F1'].mean())
print('F05 mean:', VAL['F05'].mean())

# %%
Test = Test.sort_index()

for AOI in Test.index.levels[1]:
    out = Test.loc[ix[:,AOI],:].copy()
    out = out.loc[:,out.droplevel(1, axis = 1).columns.str.startswith('pred')]\
        .droplevel(1,axis = 1).reset_index(['Gap','AOI']).drop(columns = 'AOI')

    out = out.astype(np.bool_).astype(int)

    out = out.swaplevel().sort_index(level = ['y','x'])

    ds = xr.Dataset.from_dataframe(out)\
        .sortby(['y','x'])\
            .rio.write_crs(CRS)\
                .rio.reproject(4326)

    fig, ax = plt.subplot_mosaic([['Gap','coh'],
                                  ['S1', 'coh_S1']], figsize = (12,10))

    ds['Gap'].plot(ax = ax['Gap'])
    ds['pred_coherence'].plot(ax = ax['coh'])
    ds['pred_s1'].plot(ax = ax['S1'])
    ds['pred_s1+coherence'].plot(ax = ax['coh_S1'])

    ax['Gap'].set_title('Gap ground-truth')
    ax['coh'].set_title('Pred.: coherence')
    ax['S1'].set_title('Pred.: WI =delta(VV+VH')
    ax['coh_S1'].set_title('Pred.: WI+Coherence')
    plt.show()
    #ds.rio.to_raster(ROOT + "preds/" + AOI)
