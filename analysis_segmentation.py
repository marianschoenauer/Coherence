# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 19:06:17 2025

@author: Marian Schonauer
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
#import geopandas as gpd
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

A = 'ITA'

USER = "Marian"

if USER == "Marian":
    ROOT = "D:/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/"
else:
    ROOT = "C:/Users/Lika/OneDrive - Mendelova univerzita v Brně/Coherence_VI_Krtiny/"


FIGS, TABS = ROOT + "Manuscript/figures/", ROOT + "Manuscript/tables/"

conc = xr.load_dataset(ROOT + "satellite_data/"+A+"_conc.nc", engine = "h5netcdf")
conc = conc.assign(WI = conc['VV'] + conc['VH'])
conc = conc.assign(RRVI = conc['VH']/conc['VV'])
conc = conc.assign(NDVI = (conc["B8"] - conc["B4"]) / (conc["B8"] + conc["B4"]))
#conc = conc.rio.write_crs(conc.spatial_ref.attrs['crs_wkt'])

# %% plot Time Series

sites = {
    'SLP': 'SLP_test_BYC1.tif',
    'ITA': 'ITA_test_5.tif',
    'GER': 'GER_test_5.tif',
}

# .get(A) returns None (or a default value) if A isn't found
SITE = sites.get(A)

TN = rio.open_rasterio(ROOT + "clusters/TN/"+SITE)
CRS = TN.spatial_ref.attrs['crs_wkt']

TP = rio.open_rasterio(ROOT + "clusters/TP/"+SITE)\
    .rio.reproject_match(TN)

tp = TP.where(TP != 255, np.nan)\
    .to_dataframe(name = 'PlanSco')\
                .dropna()\
                    .droplevel('band')

tn = TN.where(TN != 255, np.nan)\
    .to_dataframe(name = 'PlanSco')\
                .dropna()\
                    .droplevel('band')

sat = conc\
    .rio.reproject_match(TN)\
        .to_dataframe()\
                .swaplevel()\
                    .reset_index('time')\
                        .sort_index()

tp_sat = sat.loc[tp.index,:].assign(Gap = True)
tn_sat = sat.loc[tn.index,:].assign(Gap = False)

pl = pd.concat([tp_sat, tn_sat])

pl = pl.groupby(['time','Gap']).mean()

pl['days'] = pl.reset_index().time.dt.days.values

for col in pl.columns.drop(['spatial_ref', 'days']):
    sns.lineplot(data = pl.reset_index(), x = 'days', y = col, hue = 'Gap')
    plt.axvline(0)
    plt.savefig(FIGS + 'ts_' +A + col + '.png', dpi = 300)
    plt.show()

del SITE
# %% create Mean diffs

month_before = pd.to_timedelta(-4, unit = 'W')
day_0 = pd.to_timedelta(0, unit = 'd')
delay  = pd.to_timedelta(7, unit = 'd')
month_after = pd.to_timedelta(8 if A == 'GER' else 4, unit = 'W')

Before = conc.sel({"time":slice(month_before, day_0)}).mean(dim = 'time')
After = conc.sel({"time":slice(delay, month_after)}).mean(dim = 'time')

Mean = Before - After
ds = Mean.fillna(Mean.mean())

del month_before, month_after, delay, Before, After, Mean, conc

# %% Segmentation features

list_da = []
coords = ds.coords

for feat_name in list(ds.data_vars):
    feat = feature.multiscale_basic_features(ds[feat_name].values,
                                      intensity=True,
                                      edges=False,
                                      texture=True,
                                      sigma_min=1,
                                      sigma_max=16,
                                      num_sigma = 3)

    np_name = np.array(feat_name)
    np_sigmas = np.array(["1","2","3"])

    feat_names = np.concatenate((
        np_name + np.array(["_intensity_"])+ np_sigmas,
        np_name + np.array(["_texture1_"])+ np_sigmas,
        np_name + np.array(["_texture2_"])+ np_sigmas))

    coords_i = coords.assign(feature = feat_names)

    da_feat = xr.DataArray(feat, coords_i[['y','x','feature']].coords)
    da_feat.name = feat_name

    list_da.append(da_feat)

ds = xr.merge(list_da)

# %% labelling
df_list = []

for PATH in glob.glob(ROOT + "clusters/TN/"+A+"*.tif"):

    AOI_name = PATH.replace(ROOT + "clusters/TN\\","")
    print(AOI_name)

    TN = rio.open_rasterio(PATH)

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

    sat = ds.transpose('feature', 'y', 'x')\
        .rio.reproject_match(TN)\
            .to_dataframe()\
                .unstack(0)\
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
    .set_index(['Gap'], append = True)\
        .sort_index()

del df_list, tp_sat, tn_sat, sat, TN, TP, tp, tn

# %% set index

df = df_nc.loc[~df_nc.index.duplicated()].copy()
df = df.reset_index()\
    .set_index(['Gap', 'AOI','x', 'y'])\
        .sort_index()\
            .drop(columns = 'spatial_ref')\
                .dropna(how = 'all', axis = 1)

df = df.fillna(df.mean())

# %% settings

cols_s2 = ['NDVI','B12',] #'B3','B4','B5','B8A','B11',
cols_backscatter = ['WI', 'VV', 'RRVI'] #'Rc','RVI','VH',
cols_coherence_12 = ['coh_12'] #,
cols_coherence_12_24_36 = ['coh_12', 'coh_24', 'coh_36', ]

sce_data = {'coherence_12_24_36':cols_coherence_12_24_36,
            'coherence':cols_coherence_12,
            'backscatter':cols_backscatter,
            'backscatter+coherence':cols_backscatter+cols_coherence_12,
            's2':cols_s2,
            's2+backscatter': cols_s2 + cols_backscatter,
            's2+coherence': cols_s2 + cols_coherence_12,
            's2+backscatter+coherence': cols_s2+ cols_backscatter + cols_coherence_12
           }

sce = pd.DataFrame({'cols': sce_data.values()}, index=sce_data.keys())

del cols_s2, cols_backscatter, cols_coherence_12, cols_coherence_12_24_36, sce_data

# %% model training

output_settings = []

output_aois = []

#Partition = df.reset_index()['AOI'].isin([A+'_train.tif']).values

Tests = []
Coefs = []
for Setting, cols in sce.iterrows():

    tests = []
    coefs = []
    for test_area in df.index.levels[1][:-1]:

        Partition = df.reset_index()['AOI'].isin([test_area]).values

        train = df.loc[~Partition,cols.iloc[0]].copy()
        test = df.loc[Partition,cols.iloc[0]].copy()

        train.columns.names = ['Feature', 'feature_seg']

        if np.any(test.reset_index(['Gap','AOI']).index.isin(
                train.reset_index(['Gap','AOI']).index)):
            print("OVERLAPPING PIXELS")

        y = train.reset_index().loc[:,'Gap'].values.flatten()
        X = train.to_numpy()

        pos_weight = train.groupby('Gap').size()[False]/train.groupby('Gap').size()[True]
        weights = np.where(y, pos_weight, 1)
        """
        model = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",      # must be 'saga' for elastic net
                    l1_ratio=0.5        # 0 = ridge, 1 = lasso
                )
        """
        pipe = Pipeline([('scaler',StandardScaler()),
                         ('model',model(random_state = 1,\
                                        solver = 'svd'))]) #(random_state = 1)
        #cale_pos_weight = pos_weights

        model_fit = pipe.fit(X = X, y = y,
                             model__sample_weight = weights)
        test['pred_' + Setting] = model_fit.predict(test)

        tests.append(test.loc[:,['pred_' + Setting]])

        coef = pd.DataFrame({"weight": model_fit['model'].coef_.flatten()},
                            index = train.columns)\
            .assign(Setting = Setting, test_area = test_area)\
                .set_index(['Setting','test_area'], append = True)

        coefs.append(coef)

        del train, test, y, X, pos_weight, weights, pipe, model_fit

    Tests.append(pd.concat(tests, axis = 0))
    Coefs.append(pd.concat(coefs, axis = 0))

Test = pd.concat(Tests, axis=1)
Coef = pd.concat(Coefs, axis = 0)

Test.index = Test.index.remove_unused_levels()

# %% weights

coefs = Coef.groupby(['Feature' ,'feature_seg' , 'Setting']).mean()

cc = coefs\
        .reorder_levels([2,0,1])\
            .sort_index()

cc.unstack('Setting').to_excel(TABS + A+'model_weights.xlsx')

# %% validation metrics

results_val = []

for (aoi), group in Test.groupby('AOI'):

    y_true = group.reset_index(['Gap'])['Gap'].values

    for Setting in group.columns:
        y_pred = group.loc[:,Setting].values

        F1 = f1_score(y_true = y_true, y_pred = y_pred, zero_division = 0)
        F05 = fbeta_score(y_true = y_true, y_pred = y_pred, beta = 0.5, \
                          zero_division = 0)
        Precision = precision_score(y_true = y_true, y_pred = y_pred, \
                                    zero_division = 0)
        Recall = recall_score(y_true = y_true, y_pred = y_pred, \
                                    zero_division = 0)
        MCC = matthews_corrcoef(y_true = y_true, y_pred = y_pred)

        results_val.append({'setting':Setting[0][5:],
                        'test_area':aoi,
                        'F1':F1,
                        'F05': F05,
                        'Precision':Precision,
                        'Recall':Recall,
                        'MCC':MCC})    

VAL = pd.DataFrame(results_val).set_index(['setting','test_area'])\
    .sort_index()
VAL.columns.name = 'Metric'

sns.boxplot(VAL.stack(level = 'Metric').to_frame(name = "value"), \
            x = 'setting', y = 'value', hue = 'Metric')
plt.savefig(FIGS + A+ '_val_metrics_boxplot.png')
plt.show()

summary = VAL.groupby('setting')['F1'].describe()\
    .sort_values('mean', ascending = False)

print('F1', summary)

best_set = summary['mean'].idxmax()

VAL.to_excel(TABS+A+"val_metrics.xlsx")

# %%
for best_set in summary['mean'].index:
    pl = cc.loc[best_set,:].sort_values('weight').reset_index()

    sns.barplot(pl,
        x = 'weight',
        y = 'feature_seg'
        )
    plt.title(best_set)
    plt.show()

# %% print prediction maps
Test = Test.sort_index()

OUT = ROOT + "preds/"

os.mkdir(OUT+A)

for AOI in Test.index.levels[1]:

    OUT_sub = OUT+A+'\\'+AOI.replace('.tif','')
    os.mkdir(OUT_sub)
    out = Test.loc[ix[:,AOI],:].copy()

    #if SEG:
    out = out.loc[:,out.droplevel(1, axis = 1).columns.str.startswith('pred')]\
        .droplevel(1,axis = 1).reset_index(['Gap','AOI']).drop(columns = 'AOI')
    #else:
     #   out = out.loc[:,out.columns.str.startswith('pred')]\
      #      .reset_index(['Gap','AOI']).drop(columns = 'AOI')

    out = out.astype(np.bool_).astype(int)

    out = out.swaplevel().sort_index(level = ['y','x'])

    ds = xr.Dataset.from_dataframe(out)\
        .sortby(['y','x'])\
            .rio.write_crs(CRS)\
                .rio.reproject(4326)

    for VAR in list(ds.data_vars):
        ds[VAR].rio.to_raster(OUT_sub + '\\' + VAR + '.tif')
