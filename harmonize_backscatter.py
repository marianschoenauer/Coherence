# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:01:28 2026

@author: Marian Schonauer
"""


A, name_s1_backscatter =  'GER', 'Friedrike_backscatter_32632.nc'

s1_bs_full =  xr.load_dataset(ROOT +'satellite_data/S1_backscatter/'+name_s1_backscatter, 
                           engine = 'h5netcdf')

s1_bs_full = s1_bs_full[['VV','VH']]

s1_bs_full['VV'] = 10*np.log10(s1_bs_full['VV'])
s1_bs_full['VH'] = 10*np.log10(s1_bs_full['VH'])  

CRS = s1_bs_full.spatial_ref.attrs['crs_wkt']

s1_bs_full = s1_bs_full\
    .rio.write_crs(CRS)

s1_bs_full.to_netcdf(ROOT +'satellite_data/S1_backscatter/GER_backscatter_dB.nc', engine = 'h5netcdf')


# ITA
A, name_s1_backscatter =  'ITA', 'ITA_backscatter_stack_RTC10_10m_32633_12day_pairs.nc'

s1_bs_full =  xr.load_dataset(ROOT +'satellite_data/S1_backscatter/'+name_s1_backscatter, 
                           engine = 'h5netcdf')

s1_bs_full = s1_bs_full[['gamma0_VV','gamma0_VH']].rename({'gamma0_VV':'VV','gamma0_VH':'VH'})

s1_bs_full['VV'] = 10*np.log10(s1_bs_full['VV'])
s1_bs_full['VH'] = 10*np.log10(s1_bs_full['VH'])  

CRS = s1_bs_full.spatial_ref.attrs['crs_wkt']

s1_bs_full = s1_bs_full\
    .rio.write_crs(CRS)

s1_bs_full.to_netcdf(ROOT +'satellite_data/S1_backscatter/ITA_backscatter_dB.nc', engine = 'h5netcdf')

'''
'''
s1_bs_full =  xr.load_dataset(ROOT+'satellite_data/s1/'+A+'.nc', 
                           engine = 'h5netcdf')

CRS = s1_bs_full.spatial_ref.attrs['crs_wkt']

s1_bs_full = s1_bs_full\
    .rio.write_crs(CRS)
np.diff(s1_bs_full.x)

xmin, ymin, xmax, ymax = AOIs.to_crs(CRS).total_bounds

#s1_bs_full = s1_bs_full.sel({'x':slice(xmin, xmax), 'y':slice(ymax,ymin)})   

# Landsat
s2_full =  xr.load_dataset(ROOT + "landsat/"
                +A+".nc", engine = 'h5netcdf')

np.diff(s2_full.x)
s2_full = s2_full\
    .rio.write_crs(s2_full.spatial_ref.attrs['crs_wkt'])\
        .rio.reproject_match(s1_bs_full)
        
s2_full["NDVI"]  = (s2_full["SR_B5"] - s2_full["SR_B4"]) / (s2_full["SR_B5"] + s2_full["SR_B4"])
