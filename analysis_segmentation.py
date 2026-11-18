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
import geopandas as gpd
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
from xgboost import XGBClassifier as model
#from sklearn.preprocessing import OneHotEncoder
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

Df = pd.read_pickle(ROOT + "df.pkl")

# create country as dummy-coded variables
dummies = pd.get_dummies(Df.reset_index('Country')['Country'], prefix='C', dtype=int)

dummies.columns = pd.MultiIndex.from_arrays([list(dummies.columns), [' ', ' ', ' ']])

Df = Df.join(dummies)

# add weights for sample sizes
sizes= Df.groupby(['Gap']).size()
sizes = 1 / (sizes  / sizes.sum()).to_frame(('weights', " "))
Df = Df.merge(sizes, left_index = True, right_index = True)

Df = Df.sort_index()

Df.columns.levels[0]

TRAINING = False
SINGLE_TESTING = False

Df.groupby('Gap').size() / Df.shape[0]
# %% plot Time Series
sites = {
    'SLP': 'SLP_49.tif',
    'ITA': 'ITA_70.tif',
    'GER': 'GER_5.tif',
}

SITE = sites.get('SLP')

conc = xr.load_dataset(NC +SITE.replace('.tif','.nc'), engine = "h5netcdf")
conc = conc.rio.write_crs(conc.spatial_ref.attrs['crs_wkt'])

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
                        .sort_index()\
                            .drop(columns = ['spatial_ref'])

tp_sat = sat.loc[tp.index,:].assign(Gap = True)
tn_sat = sat.loc[tn.index,:].assign(Gap = False)

pl = pd.concat([tp_sat, tn_sat])

pl = pl.groupby(['time','Gap']).mean()

pl['days'] = pl.reset_index().time.dt.days.values

for col in pl.columns.drop(['days']):
    sns.lineplot(data = pl.reset_index(), x = 'days', y = col, hue = 'Gap')
    plt.axvline(0)
    plt.show()

del SITE, col, conc, pl, sat, sites, TN, tn, tn_sat, TP, tp_sat, tp

# %% settings

if SINGLE_TESTING:
    single = ['B4', 'B8','B12', 
               'EVI2', 'NBR', 'NDVI', 
               'RNDVI', 'RRVI', 'RVI', 
               'SDVI', 'SSVI', 'VH', 'VV',
               'coh12',]
    sce_data = {**{f'{col}_min': [f'{col}_min'] for col in single},
                **{f'{col}_max': [f'{col}_max'] for col in single},
                **{f'{col}': [f'{col}_min', f'{col}_max'] for col in single}
                }

else:

    single_opt = ['B8', 'B12', 'NDVI', 'NBR']
    single_bac = ['VV', 'VH', 'SSVI', 'RVI']
    single_coh = ['coh12']
    
    single = single_opt + single_bac + single_coh

    cols_opt = [f'{col}_min' for col in single_opt] + [f'{col}_max' for col in single_opt]
    
    cols_bac = [f'{col}_min' for col in single_bac] + [f'{col}_max' for col in single_bac]
    
    cols_coh = [f'{col}_min' for col in single_coh] + [f'{col}_max' for col in single_coh]
    
    sce_data = {
                
                #**{f'{col}': [f'{col}_min', f'{col}_max'] for col in single},
                
                'coherence': cols_coh,
    
                'backscatter': cols_bac,
                'backscatter+coherence': cols_bac+cols_coh,
                #'backscatter+coherence40': cols_backscatter+['coh40_12_min'],
                'optical': cols_opt,
                'optical+backscatter': cols_opt + cols_bac,
                'optical+coherence': cols_opt + cols_coh,
                #'s2+coherence40': cols_s2 + ['coh40_12_min'],
                'optical+backscatter+coherence': cols_opt+ cols_bac + cols_coh,
                #'s2+backscatter+coherence40': cols_s2+ cols_backscatter + ['coh40_12_min']
               }

Country = ['C_SLP', 'C_GER', 'C_ITA']

sce_data = {
    name: cols + Country
    for name, cols in sce_data.items()
}

sce = pd.DataFrame({'cols': sce_data.values()}, index=sce_data.keys())

sce.to_excel(TABS + 'individual_features.xlsx' if SINGLE_TESTING else 'settings_reduced.xlsx')

# %% check collinearity between features
if SINGLE_TESTING:

    corr_matrix = Df.loc[:,[ (f'{col}_max',f'{col}_max_intensity_1') for col in single]].corr()
    
    # Set up the plotting environment
    plt.figure(figsize=(10, 8))
    
    # Draw the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(FIGS + 'corrMatrix.png')
    plt.show()

# %% model training
if TRAINING:
    output_settings = []
    output_aois = []
    
    #Partition = df.reset_index()['AOI'].isin([A+'_train.tif']).values
    #Df = Df.loc[['SLP'],:]
    
    Tests = []
    Coefs = []
    for Setting, cols in sce.iterrows():#
        print(Setting)
    
        tests = []
        coefs = []
        for test_area in Df.index.remove_unused_levels().levels[2]:
            
            Partition = Df.reset_index('AOI')['AOI'].isin([test_area]).values
    
            train = Df.loc[~Partition,cols.iloc[0]].copy()
            weights = Df.loc[~Partition,'weights'].values
            test = Df.loc[Partition,cols.iloc[0]].copy()
    
            train.columns.names = ['Feature', 'feature_seg']
    
            '''
            if np.any(test.reset_index(['Gap','AOI']).index.isin(
                    train.reset_index(['Gap','AOI']).index)):
                print("OVERLAPPING PIXELS")
            '''
            
            y = train.reset_index().loc[:,'Gap'].values.flatten()
            X = train.to_numpy()
    
            pipe = Pipeline([('scaler',StandardScaler()),
                             ('model',model(random_state = 0))])
    
            model_fit = pipe.fit(X = X, y = y, model__sample_weight = weights)
            test['pred_' + Setting] = model_fit.predict(test)
    
            tests.append(test.loc[:,['pred_' + Setting]])
            
            coef = pd.DataFrame({"weight": model_fit['model'].feature_importances_.flatten()},
                                index = train.columns)\
                .assign(Setting = Setting, fold = test_area)\
                    .set_index(['Setting','fold'], append = True)
    
            coefs.append(coef)
    
            del train, test, y, X,  weights, pipe, model_fit
    
        Tests.append(pd.concat(tests, axis = 0))
        Coefs.append(pd.concat(coefs, axis = 0))
    
    Test = pd.concat(Tests, axis=1)
    Coef = pd.concat(Coefs, axis = 0)
    
    Test.index = Test.index.remove_unused_levels()
    
    # feature importances
    
    coefs = Coef.groupby(['Feature' ,'feature_seg' , 'Setting', 'fold']).mean()\
            .reorder_levels([2,0,1,3])\
                .sort_index()
                
    #  validation metrics
    
    results_val = []
    
    for (aoi), group in Test.groupby('AOI'):
    
        y_true = group.reset_index(['Gap'])['Gap'].values
    
        for Setting in group.columns:
            y_pred = group.loc[:,Setting].values
    
            F1 = f1_score(y_true = y_true, y_pred = y_pred, zero_division = 0)
            Precision = precision_score(y_true = y_true, y_pred = y_pred, \
                                        zero_division = 0)
            Recall = recall_score(y_true = y_true, y_pred = y_pred, \
                                        zero_division = 0)
    
            results_val.append({'setting':Setting[0][5:],
                            'test_area':aoi,
                            'F1':F1,
                            'Precision':Precision,
                            'Recall':Recall})    
    
    VAL = pd.DataFrame(results_val).set_index(['setting','test_area'])\
        .sort_index()
    VAL.columns.name = 'Metric'
    
    VAL['Country'] = VAL.reset_index('test_area')['test_area'].str[:3].values
    VAL = VAL.set_index('Country',append = True)
    
    VAL.to_pickle(ROOT + 'Objects/VAL.pkl')
    Test.to_pickle(ROOT + 'Objects/Test.pkl')
    coefs.to_pickle(ROOT + 'Objects/coefs.pkl')
# %%

VAL = pd.read_pickle(ROOT + 'Objects/VAL.pkl')
Test = pd.read_pickle(ROOT + 'Objects/Test.pkl')
coefs = pd.read_pickle(ROOT + 'Objects/coefs.pkl')

sns.boxplot(VAL, \
            y = 'setting', x = 'F1', hue = 'Country', 
            order = sce.index)
plt.savefig(FIGS + 'LOAO_ind_val_metrics_boxplot.png')
plt.show()

summary_countries = VAL.groupby(['Country','setting'])['F1'].describe()
print('F1', summary_countries)

summary = VAL.groupby(['setting'])['F1'].describe().sort_values(by = 'mean')
print('F1', summary)

best_set = summary['mean'].idxmax()
# %% plot 

fig, ax = plt.subplot_mosaic([['SLP','GER', 'ITA']])

for Cou in ['SLP','GER', 'ITA']:
    
    sns.boxplot(data = VAL.loc[ix[:,:,Cou], :].stack().to_frame('value'), 
                y = 'setting', x = 'value', ax = ax[Cou],
                hue = 'Metric', legend = True if Cou == 'GER' else False)
    
    if Cou != 'SLP':
        ax[Cou].set_yticklabels([])
        ax[Cou].set_ylabel(None)


# %% print to tables

coefs.unstack(['Setting', 'Feature', 'feature_seg'])\
    .to_excel(TABS + 'LOAO_model_weights_7daysMinMax_individual_features.xlsx' if SINGLE_TESTING\
                                   else TABS + 'LOAO_model_weights_7daysMinMax_reduced.xlsx')
VAL.reset_index().to_excel(TABS + 'LOAO_val_metrics_7daysMinMax_individual_features.xlsx' if SINGLE_TESTING\
                                   else TABS + 'LOAO_val_metrics_7daysMinMax_reduced.xlsx')
summary_countries.loc[:,['mean','std']].to_excel(TABS + 'LOAO_summary_countries_7daysMinMax_individual_features.xlsx' if SINGLE_TESTING\
                                   else TABS + 'LOAO_summary_7daysMinMax_reduced.xlsx')
summary.to_excel(TABS + 'LOAO_summary_countries_7daysMinMax_individual_features.xlsx' if SINGLE_TESTING\
                                   else TABS + 'LOAO_summary_7daysMinMax_reduced.xlsx')
#  print prediction maps
Test = Test.sort_index()

for VAR in ['F1', 'Precision', 'Recall']:
    
    out = VAL.groupby(['Country','setting'])[VAR].describe()\
        .loc[:,['mean','std']].unstack(['Country']).round(3)\
            .swaplevel(axis = 1).sort_index(axis = 1)
    out.to_excel(TABS + 'summary_' + VAR + '.xlsx')
    print('\n', VAR, '\n', '\n', out)

# %% coefs

bestC = pd.read_pickle(ROOT + 'Objects/coefs.pkl').loc[best_set, :]

fig, ax = plt.subplots(1,2, tight_layout = True, figsize = (8,4))
sns.barplot(data = bestC, x = 'weight', y = 'Feature', ax = ax[0])
ax[0].set_xscale('log')

bestC.groupby(['Feature']).mean().sort_values('weight')


bestC['segmentation_type'] = np.where(bestC.index.get_level_values('feature_seg').str.contains('texture1'), 'texture1', 
                                      np.where(bestC.index.get_level_values('feature_seg').str.contains('texture2'), 'texture2', 
                                               np.where(bestC.index.get_level_values('feature_seg').str.contains('intensity'), 'intensity',pd.NA 
                                                                                     )))

bestC = bestC.reset_index()\
    .dropna(subset = 'segmentation_type')

sns.heatmap(bestC.groupby(['Feature','segmentation_type'])['weight'].mean().unstack('segmentation_type'), ax = ax[1])
plt.savefig(FIGS + "featureImportance.png")

# %% results
from scipy.stats import ttest_rel

summary.loc[ix[:,['mean','std']]].sort_values(by = 'mean')
# overall
a = VAL.loc['optical', 'F1']
b = VAL.loc['optical+coherence', 'F1']
c = VAL.loc['optical+backscatter', 'F1']
d = VAL.loc['optical+backscatter+coherence', 'F1']

ttest_rel(a=a, b = b)
ttest_rel(a=a, b = c)
ttest_rel(a=a, b = d)


opt = VAL.loc[ix['optical',:,'SLP'],'F1']
optBa = VAL.loc[ix['optical+backscatter',:,'SLP'],'F1']
optBaCo = VAL.loc[ix['optical+backscatter+coherence',:,'SLP'],'F1']

ttest_rel(a = opt, b = optBa)
ttest_rel(a = opt, b = optBaCo)
ttest_rel(a = optBa, b = optBaCo)


opt = VAL.loc[ix['optical',:,'GER'],'F1']
optBa = VAL.loc[ix['optical+coherence',:,'GER'],'F1']
optCo = VAL.loc[ix['coherence',:,'GER'],'F1']

ttest_rel(a = opt, b = optBa)
ttest_rel(a = opt, b = optBa)
ttest_rel(a = optBa, b = optCo)

# %% F1 vs. size

# merge gaps

gaps = []
for layer in ["canopy_diffs", "Italy_windthrows", "Germany_Friederike"]:
    gaps.append(gpd.read_file(ROOT + "shapefiles/gaps.gpkg", layer = layer).to_crs(3035))
    
gaps = pd.concat(gaps)
gaps['Area'] = gaps.area

AOIs = gpd.read_file(ROOT + "shapefiles/gaps.gpkg", layer = "AOIs_individual_gaps_3035")
AOIs['AOI'] = AOIs['Country'] + "_" + AOIs['ID_No'].astype(str)

AOIs = AOIs.set_index('AOI')

AOIs['Area'] =  AOIs.area/10000

AOIs.groupby('Country')['Area'].describe()

sizes = AOIs.sjoin(gaps.loc[:,['geometry','Area']])

sizes = sizes.groupby('AOI').sum('Area')

sizes.index.names = ['test_area']

sizes = VAL.merge(sizes, left_index = True, right_index = True)

sns.scatterplot(data = sizes.loc[best_set,:], x = 'Area', y = 'F1', hue = 'Country')
plt.xscale('log')
plt.xlabel('area [1000 m2]')

plt.savefig(FIGS + 'F1_vs_gapsize.png')
plt.show()

sns.boxplot(data = sizes, x = 'Country', y = 'Area')
plt.yscale('log')
plt.ylabel('Gap size [m²]')
plt.xlabel('')
plt.savefig(FIGS + 'Gapsize.png')
# %% rasters
OUT = ROOT + "preds/"

for AOI in Test.index.levels[2]:

    OUT_sub = OUT+'LOAO/'+AOI.replace('.tif','')
    #os.mkdir(OUT_sub)
    out = Test.loc[ix[:,:,AOI],:].copy()

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
                .rio.reproject(3035)

    for VAR in list(ds.data_vars):
        ds[VAR].rio.to_raster(OUT_sub + '/' + VAR + '.tif')
