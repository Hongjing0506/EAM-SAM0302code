'''
Author: ChenHJ
Date: 2022-04-23 12:49:42
LastEditors: ChenHJ
LastEditTime: 2022-04-24 17:11:01
FilePath: /chenhj/0302code/cal_EUTT_IUTT_regress.py
Aim: 
Mission: 
'''
# %%
import numpy as np
import xarray as xr
import os
import re
from cdo import Cdo
import shutil
import sys

sys.path.append("/home/ys17-23/chenhj/self_def/")
import plot as sepl
import cal as ca
import pandas as pd
from importlib import reload

import metpy.calc as mpcalc
import metpy.xarray as mpxr
import metpy.constants as constants
import geocat.comp
from windspharm.xarray import VectorWind

reload(sepl)

# sd.path.append("/home/ys17-23/chenhj/1201code/self_def.py")

cdo = Cdo()

# for plot
import proplot as pplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter
from cartopy.mpl.ticker import LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t
from scipy import signal
from eofs.multivariate.standard import MultivariateEof
from eofs.standard import Eof
from statsmodels.distributions.empirical_distribution import ECDF
import dask


def patches(ax, x0, y0, width, height, proj):
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x0, y0), width, height, fc="none", ec="grey7", linewidth=0.8, zorder=1.1, transform=proj, linestyle="--",
    )
    ax.add_patch(rect)
# %%
#   read the data
#   read the ERA5 reanalysis data
fhgtERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/hgt_mon_r144x72_195001-201412.nc"
)
hgtERA5 = fhgtERA5["z"]


fuERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc"
)
uERA5 = fuERA5["u"]

fvERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/vwind_mon_r144x72_195001-201412.nc"
)
vERA5 = fvERA5["v"]

ftERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/temp_mon_r144x72_195001-201412.nc")
tERA5 = ftERA5["t"]

fspERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/sp_mon_r144x72_195001-201412.nc"
)
spERA5 = fspERA5["sp"]

#   read the precipitation data
fpreCRU = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]
preCRU = fpreCRU["pre"]
preCRU_JJA = ca.p_time(preCRU, 6, 8, True)/30.67
preCRU_JJA = ca.detrend_dim(preCRU_JJA, "time", deg=1, demean=False)
preCRU_JJA.attrs["units"] = "mm/day"

# GPCP data just have 1979-2014 year
fpreGPCP = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]
preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)
preGPCP_JJA = ca.detrend_dim(preGPCP_JJA, "time", deg=1, demean=False)

#   historical and ssp585 data

fhgthis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/zg_historical_r144x72_195001-201412.nc")
hgthis_ver_JJA = fhgthis_ver_JJA["zg"]
# hgthis_ver_JJA = hgthis_ver_JJA - hgthis_ver_JJA.mean(dim="lon", skipna=True)

fuhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ua_historical_r144x72_195001-201412.nc")
uhis_ver_JJA = fuhis_ver_JJA["ua"]

fvhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/va_historical_r144x72_195001-201412.nc")
vhis_ver_JJA = fvhis_ver_JJA["va"]

fthis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ta_historical_r144x72_195001-201412.nc")
this_ver_JJA = fthis_ver_JJA["ta"]

fprehis = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pr_historical_r144x72_195001-201412.nc")
prehis_JJA = fprehis["pr"]
prehis_JJA.attrs["units"] = "mm/day"
prehis_JJA.attrs["standard_name"] = "precipitation"

fsphis_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ps_historical_r144x72_195001-201412.nc")
sphis_JJA = fsphis_JJA["ps"]

fhgtssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/zg_ssp585_r144x72_201501-209912.nc")
hgtssp585_ver_JJA = fhgtssp585_ver_JJA["zg"]
# hgtssp585_ver_JJA = hgtssp585_ver_JJA - hgtssp585_ver_JJA.mean(dim="lon", skipna=True)

fussp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ua_ssp585_r144x72_201501-209912.nc")
ussp585_ver_JJA = fussp585_ver_JJA["ua"]

fvssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/va_ssp585_r144x72_201501-209912.nc")
vssp585_ver_JJA = fvssp585_ver_JJA["va"]

ftssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ta_ssp585_r144x72_201501-209912.nc")
tssp585_ver_JJA = ftssp585_ver_JJA["ta"]

fpressp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pr_ssp585_r144x72_201501-209912.nc")
pressp585_JJA = fpressp585["pr"]
pressp585_JJA.attrs["units"] = "mm/day"
pressp585_JJA.attrs["standard_name"] = "precipitation"

fspssp585_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ps_ssp585_r144x72_201501-209912.nc")
spssp585_JJA = fspssp585_JJA["ps"]


# %%
hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True).loc[:, 100.0:, :, :]
# hgtERA5_ver_JJA = hgtERA5_ver_JJA - hgtERA5_ver_JJA.mean(dim="lon", skipna=True)
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, :, :]
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, :, :]
tERA5_ver_JJA = ca.p_time(tERA5, 6, 8, True).loc[:, 100.0:, :, :]
preCRU_JJA = ca.p_time(preCRU, 6, 8, True) / 30.67
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)

hgtERA5_ver_JJA = ca.detrend_dim(hgtERA5_ver_JJA, "time", deg=1, demean=False)
uERA5_ver_JJA = ca.detrend_dim(uERA5_ver_JJA, "time", deg=1, demean=False)
vERA5_ver_JJA = ca.detrend_dim(vERA5_ver_JJA, "time", deg=1, demean=False)
tERA5_ver_JJA = ca.detrend_dim(tERA5_ver_JJA, "time", deg=1, demean=False)
preCRU_JJA = ca.detrend_dim(preCRU_JJA, "time", deg=1, demean=False)
spERA5_JJA = ca.detrend_dim(spERA5_JJA, "time", deg=1, demean=False)

preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)
preGPCP_JJA = ca.detrend_dim(preGPCP_JJA, "time", deg=1, demean=False)
# %%
#   calculate the historical dsdpg between 200hPa to 500hPa
ptop = 1 * 200
g = 9.8
his_dslevel = hgthis_ver_JJA.coords["level"].loc[500.0:200.0] * 100.0
his_dslevel.attrs["units"] = "Pa"
# his_dsdp = geocat.comp.dpres_plevel(his_dslevel, sphis_JJA, ptop)
# print(sphis_ds_JJA)
his_dsdp = xr.apply_ufunc(
    geocat.comp.dpres_plevel,
    his_dslevel,
    sphis_JJA,
    ptop,
    input_core_dims=[["level"], [], []],
    output_core_dims=[["level"]],
    vectorize=True,
    dask="parallelized",
)
# # for i in np.arange(0,26):
# #     print(his_dsdp[i, 0, 0, 0, :])
his_dsdp = his_dsdp.transpose("models", "time", "level", "lat", "lon")
his_dsdpg = his_dsdp / g
his_dsdpg.attrs["units"] = "kg/m2"
his_dsdpg.name = "dsdpg"
his_dsdpg.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_dsdpg500-200.nc")

#   calculate the historical vertical intergrated temperature

utthis_JJA = (this_ver_JJA.loc[:,:,500.0:200.0,:,:] * his_dsdpg.data).sum(dim="level", skipna=True)
utthis_JJA.name = "utt"
# uq_dpg_his_JJA = ca.detrend_dim(uq_dpg_his_JJA, "time", deg=1, demean=False)
# uq_dpg_his_JJA.attrs["units"] = "100kg/(m*s)"
utthis_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_utt_500-200hPa.nc")

# %%
#   calculate the ssp585 dsdpg between 200hPa to 500hPa
ptop = 1 * 200
g = 9.8
ssp585_dslevel = hgtssp585_ver_JJA.coords["level"].loc[500.0:200.0] * 100.0
ssp585_dslevel.attrs["units"] = "Pa"
# ssp585_dsdp = geocat.comp.dpres_plevel(ssp585_dslevel, spssp585_JJA, ptop)
# print(spssp585_ds_JJA)
ssp585_dsdp = xr.apply_ufunc(
    geocat.comp.dpres_plevel,
    ssp585_dslevel,
    spssp585_JJA,
    ptop,
    input_core_dims=[["level"], [], []],
    output_core_dims=[["level"]],
    vectorize=True,
    dask="parallelized",
)
# # for i in np.arange(0,26):
# #     print(ssp585_dsdp[i, 0, 0, 0, :])
ssp585_dsdp = ssp585_dsdp.transpose("models", "time", "level", "lat", "lon")
ssp585_dsdpg = ssp585_dsdp / g
ssp585_dsdpg.attrs["units"] = "kg/m2"
ssp585_dsdpg.name = "dsdpg"
ssp585_dsdpg.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_dsdpg500-200.nc")
# %%
#   calculate the ssp585 vertical intergrated temperature

uttssp585_JJA = (tssp585_ver_JJA.loc[:,:,500.0:200.0,:,:] * ssp585_dsdpg.data).sum(dim="level", skipna=True)
uttssp585_JJA.name = "utt"
# uq_dpg_ssp585_JJA = ca.detrend_dim(uq_dpg_ssp585_JJA, "time", deg=1, demean=False)
# uq_dpg_ssp585_JJA.attrs["units"] = "100kg/(m*s)"
uttssp585_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_utt_500-200hPa.nc")
# %%
# %%
#   read the dsdpg and utt in historical run and ssp585 run
his_dsdpg = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_dsdpg500-200.nc")
ssp585_dsdpg = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_dsdpg500-200.nc")

futthis_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_utt_500-200hPa.nc")
utthis_JJA = futthis_JJA["utt"]

futtssp585_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_utt_500-200hPa.nc")
uttssp585_JJA = futtssp585_JJA["utt"]
# %%
#   calculate the detrended utt in historical and ssp585 run
utthis_JJA_detrend = ca.detrend_dim(utthis_JJA, "time", deg=1, demean=False)
uttssp585_JJA_detrend = ca.detrend_dim(uttssp585_JJA, "time", deg=1, demean=False)
utthis_JJA_detrend.name = "utt"
uttssp585_JJA_detrend.name = "utt"
utthis_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_utt_500-200hPa.nc")
uttssp585_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_utt_500-200hPa.nc")
# %%
#   calculate and output the EUTT-IUTT in historical and ssp585 run
lat = utthis_JJA.coords["lat"]
lon = utthis_JJA.coords["lon"]

lat_EUTT_range = lat[(lat >= 20.0) & (lat <= 40.0)]
lon_EUTT_range = lon[(lon >= 60.0) & (lon <= 100.0)]

lat_IUTT_range = lat[(lat >= -10.0) & (lat <= 10.0)]
lon_IUTT_range = lon[(lon >= 60.0) & (lon <= 100.0)]

EIMTGhis_JJA = ca.cal_lat_weighted_mean(utthis_JJA.sel(lat=lat_EUTT_range,lon=lon_EUTT_range)).mean(dim="lon",skipna=True)-ca.cal_lat_weighted_mean(utthis_JJA.sel(lat=lat_IUTT_range,lon=lon_IUTT_range)).mean(dim="lon",skipna=True)
EIMTGhis_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_EIMTG.nc")

EIMTGssp585_JJA = ca.cal_lat_weighted_mean(uttssp585_JJA.sel(lat=lat_EUTT_range,lon=lon_EUTT_range)).mean(dim="lon",skipna=True)-ca.cal_lat_weighted_mean(uttssp585_JJA.sel(lat=lat_IUTT_range,lon=lon_IUTT_range)).mean(dim="lon",skipna=True)
EIMTGssp585_JJA.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_EIMTG.nc")

# %%
#   calculate the detrended EIMTG in historical and ssp585 run
EIMTGhis_JJA_detrend = ca.detrend_dim(EIMTGhis_JJA, "time", deg=1, demean=False)
EIMTGssp585_JJA_detrend = ca.detrend_dim(EIMTGssp585_JJA, "time", deg=1, demean=False)
EIMTGhis_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_EIMTG.nc")
EIMTGssp585_JJA_detrend.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_EIMTG.nc")

# %%
#   calculate the EUTT-IUTT in ERA5
ptop = 1 * 200
g = 9.8
ERA5_dslevel = uERA5_ver_JJA.coords["level"].loc[200.0:500.0] * 100.0
ERA5_dslevel.attrs["units"] = "Pa"
ERA5_dsdp = geocat.comp.dpres_plevel(ERA5_dslevel, spERA5_JJA, ptop)
ERA5_dsdpg = ERA5_dsdp / g
ERA5_dsdpg.attrs["units"] = "kg/m2"
ERA5_dsdpg.name = "dsdpg"

uttERA5_JJA = (tERA5_ver_JJA.loc[:,200.0:500.0,:,:] * ERA5_dsdpg.data).sum(dim="level", skipna=True)
uttERA5_JJA.name = "utt"

EIMTGERA5_JJA = ca.cal_lat_weighted_mean(uttERA5_JJA.sel(lat=lat_EUTT_range,lon=lon_EUTT_range)).mean(dim="lon",skipna=True)-ca.cal_lat_weighted_mean(uttERA5_JJA.sel(lat=lat_IUTT_range,lon=lon_IUTT_range)).mean(dim="lon",skipna=True)
EIMTGERA5_JJA_detrend = ca.detrend_dim(EIMTGERA5_JJA, "time", deg=1, demean=False)
# %%
#   calculate the 200hPa to 500hPa thickness difference in Europe and Indian Ocean in ERA5, historical and ssp585 run
EIthdiffERA5_JJA = ca.cal_lat_weighted_mean((hgtERA5_ver_JJA.sel(level=200,lat=lat_EUTT_range,lon=lon_EUTT_range)-hgtERA5_ver_JJA.sel(level=500.0,lat=lat_EUTT_range,lon=lon_EUTT_range))).mean(dim="lon",skipna=True)-ca.cal_lat_weighted_mean((hgtERA5_ver_JJA.sel(level=200,lat=lat_IUTT_range,lon=lon_IUTT_range)-hgtERA5_ver_JJA.sel(level=500.0,lat=lat_IUTT_range,lon=lon_IUTT_range))).mean(dim="lon",skipna=True)

EIthdiffhis_JJA = ca.cal_lat_weighted_mean((hgthis_ver_JJA.sel(level=200,lat=lat_EUTT_range,lon=lon_EUTT_range)-hgthis_ver_JJA.sel(level=500.0,lat=lat_EUTT_range,lon=lon_EUTT_range))).mean(dim="lon",skipna=True)-ca.cal_lat_weighted_mean((hgthis_ver_JJA.sel(level=200,lat=lat_IUTT_range,lon=lon_IUTT_range)-hgthis_ver_JJA.sel(level=500.0,lat=lat_IUTT_range,lon=lon_IUTT_range))).mean(dim="lon",skipna=True)

EIthdiffssp585_JJA = ca.cal_lat_weighted_mean((hgtssp585_ver_JJA.sel(level=200,lat=lat_EUTT_range,lon=lon_EUTT_range)-hgtssp585_ver_JJA.sel(level=500.0,lat=lat_EUTT_range,lon=lon_EUTT_range))).mean(dim="lon",skipna=True)-ca.cal_lat_weighted_mean((hgtssp585_ver_JJA.sel(level=200,lat=lat_IUTT_range,lon=lon_IUTT_range)-hgtssp585_ver_JJA.sel(level=500.0,lat=lat_IUTT_range,lon=lon_IUTT_range))).mean(dim="lon",skipna=True)

EIthdiffERA5_JJA_detrend = ca.detrend_dim(EIthdiffERA5_JJA, "time", deg=1, demean=True)

EIthdiffhis_JJA_detrend = ca.detrend_dim(EIthdiffhis_JJA, "time", deg=1, demean=True)

EIthdiffssp585_JJA_detrend = ca.detrend_dim(EIthdiffssp585_JJA, "time", deg=1, demean=True)
# %%
#   calculate the correlation coefficients between EIMTG and EIthdiff
print(stats.linregress(EIMTGERA5_JJA_detrend, EIthdiffERA5_JJA_detrend))
print(ca.dim_linregress(EIMTGhis_JJA_detrend, EIthdiffhis_JJA_detrend)[2])
# %%
#   calculate the IWF in ERA5 and read the IWF in historical and ssp585 run
ERA5_IWF_index = ca.IWF(uERA5_ver_JJA, vERA5_ver_JJA)
ERA5_IWF_index = ca.detrend_dim(ERA5_IWF_index, "time", deg=1, demean=False)
ERA5_SAM_index = ca.SAM(vERA5_ver_JJA)
ERA5_SAM_index = ca.detrend_dim(ERA5_SAM_index, "time", deg=1, demean=False)
fhis_IWF_index = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_IWF_index_1950-2014.nc")
his_IWF_index = fhis_IWF_index["IWF"]

fssp585_IWF_index = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_IWF_index_2015-2099.nc")
ssp585_IWF_index = fssp585_IWF_index["IWF"]
# %%
#   calculate the correlation between IWF and EIMTG
(
    IWF_his_EIMTG_slope,
    IWF_his_EIMTG_intercept,
    IWF_his_EIMTG_rvalue,
    IWF_his_EIMTG_pvalue,
    IWF_his_EIMTG_hypothesis,
) = ca.dim_linregress(his_IWF_index, EIMTGhis_JJA_detrend)

(
    IWF_ssp585_p3_EIMTG_slope,
    IWF_ssp585_p3_EIMTG_intercept,
    IWF_ssp585_p3_EIMTG_rvalue,
    IWF_ssp585_p3_EIMTG_pvalue,
    IWF_ssp585_p3_EIMTG_hypothesis,
) = ca.dim_linregress(ssp585_IWF_index.sel(time=ssp585_IWF_index.time.dt.year>=2064), EIMTGssp585_JJA_detrend.sel(time=EIMTGssp585_JJA_detrend.time.dt.year>=2064))
# %%
#   calculate the delatmean hgt
hgtERA5_ver_JJA_delatmean = hgtERA5_ver_JJA - hgtERA5_ver_JJA.mean(dim="lon", skipna=True)
hgthis_ver_JJA_delatmean = hgthis_ver_JJA - hgthis_ver_JJA.mean(dim="lon", skipna=True)
hgtssp585_ver_JJA_delatmean = hgtssp585_ver_JJA - hgtssp585_ver_JJA.mean(dim="lon", skipna=True)
# %%
#   calculate the hgt,u,v regression onto EIMTG
(
    EIMTG_ERA5_hgt_slope,
    EIMTG_ERA5_hgt_intercept,
    EIMTG_ERA5_hgt_rvalue,
    EIMTG_ERA5_hgt_pvalue,
    EIMTG_ERA5_hgt_hypothesis,
) = ca.dim_linregress(EIMTGERA5_JJA_detrend, hgtERA5_ver_JJA_delatmean.sel(level=[200.0, 500.0, 850.0]))

(
    EIMTG_his_hgt_slope,
    EIMTG_his_hgt_intercept,
    EIMTG_his_hgt_rvalue,
    EIMTG_his_hgt_pvalue,
    EIMTG_his_hgt_hypothesis,
) = ca.dim_linregress(EIMTGhis_JJA_detrend, hgthis_ver_JJA_delatmean.sel(level=[200.0, 500.0, 850.0]))

(
    EIMTG_ssp585_p3_hgt_slope,
    EIMTG_ssp585_p3_hgt_intercept,
    EIMTG_ssp585_p3_hgt_rvalue,
    EIMTG_ssp585_p3_hgt_pvalue,
    EIMTG_ssp585_p3_hgt_hypothesis,
) = ca.dim_linregress(EIMTGssp585_JJA_detrend.sel(time=EIMTGssp585_JJA_detrend.time.dt.year>=2064), hgtssp585_ver_JJA_delatmean.sel(time=hgtssp585_ver_JJA_delatmean.time.dt.year>=2064, level=[200.0, 500.0, 850.0]))

(
    EIMTG_ERA5_u_slope,
    EIMTG_ERA5_u_intercept,
    EIMTG_ERA5_u_rvalue,
    EIMTG_ERA5_u_pvalue,
    EIMTG_ERA5_u_hypothesis,
) = ca.dim_linregress(EIMTGERA5_JJA_detrend, uERA5_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    EIMTG_his_u_slope,
    EIMTG_his_u_intercept,
    EIMTG_his_u_rvalue,
    EIMTG_his_u_pvalue,
    EIMTG_his_u_hypothesis,
) = ca.dim_linregress(EIMTGhis_JJA_detrend, uhis_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    EIMTG_ssp585_p3_u_slope,
    EIMTG_ssp585_p3_u_intercept,
    EIMTG_ssp585_p3_u_rvalue,
    EIMTG_ssp585_p3_u_pvalue,
    EIMTG_ssp585_p3_u_hypothesis,
) = ca.dim_linregress(EIMTGssp585_JJA_detrend.sel(time=EIMTGssp585_JJA_detrend.time.dt.year>=2064), ussp585_ver_JJA.sel(time=ussp585_ver_JJA.time.dt.year>=2064, level=[200.0, 500.0, 850.0]))

(
    EIMTG_ERA5_v_slope,
    EIMTG_ERA5_v_intercept,
    EIMTG_ERA5_v_rvalue,
    EIMTG_ERA5_v_pvalue,
    EIMTG_ERA5_v_hypothesis,
) = ca.dim_linregress(EIMTGERA5_JJA_detrend, vERA5_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    EIMTG_his_v_slope,
    EIMTG_his_v_intercept,
    EIMTG_his_v_rvalue,
    EIMTG_his_v_pvalue,
    EIMTG_his_v_hypothesis,
) = ca.dim_linregress(EIMTGhis_JJA_detrend, vhis_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    EIMTG_ssp585_p3_v_slope,
    EIMTG_ssp585_p3_v_intercept,
    EIMTG_ssp585_p3_v_rvalue,
    EIMTG_ssp585_p3_v_pvalue,
    EIMTG_ssp585_p3_v_hypothesis,
) = ca.dim_linregress(EIMTGssp585_JJA_detrend.sel(time=EIMTGssp585_JJA_detrend.time.dt.year>=2064), vssp585_ver_JJA.sel(time=vssp585_ver_JJA.time.dt.year>=2064, level=[200.0, 500.0, 850.0]))
# %%
#   ouput the correlation coefficient result into .nc files
models = EIMTG_his_hgt_slope.coords["models"]
level = EIMTG_his_hgt_slope.coords["level"]
lat = EIMTG_his_hgt_slope.coords["lat"]
lon = EIMTG_his_hgt_slope.coords["lon"]
EIMTG_his_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], EIMTG_his_hgt_slope.data),
        intercept=(["models", "level", "lat", "lon"], EIMTG_his_hgt_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], EIMTG_his_hgt_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], EIMTG_his_hgt_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], EIMTG_his_hgt_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of multi-models in historical run regress onto his_EIMTG_index"),
)
EIMTG_his_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], EIMTG_his_u_slope.data),
        intercept=(["models", "level", "lat", "lon"], EIMTG_his_u_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], EIMTG_his_u_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], EIMTG_his_u_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], EIMTG_his_u_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of multi-models in historical run regress onto his_EIMTG_index"),
)
EIMTG_his_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], EIMTG_his_v_slope.data),
        intercept=(["models", "level", "lat", "lon"], EIMTG_his_v_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], EIMTG_his_v_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], EIMTG_his_v_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], EIMTG_his_v_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of multi-models in historical run regress onto his_EIMTG_index"),
)
EIMTG_ssp585_p3_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_hgt_slope.data),
        intercept=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_hgt_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_hgt_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_hgt_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_hgt_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of multi-models in ssp585_p3 run regress onto ssp585_p3_EIMTG_index"),
)
EIMTG_ssp585_p3_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_u_slope.data),
        intercept=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_u_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_u_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_u_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_u_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of multi-models in ssp585_p3 run regress onto ssp585_p3_EIMTG_index"),
)
EIMTG_ssp585_p3_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_v_slope.data),
        intercept=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_v_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_v_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_v_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], EIMTG_ssp585_p3_v_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of multi-models in ssp585_p3 run regress onto ssp585_p3_EIMTG_index"),
)
EIMTG_his_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/EIMTG_his_hgt_regress.nc")
EIMTG_his_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/EIMTG_his_u_regress.nc")
EIMTG_his_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/EIMTG_his_v_regress.nc")
EIMTG_ssp585_p3_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/EIMTG_ssp585_p3_hgt_regress.nc")
EIMTG_ssp585_p3_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/EIMTG_ssp585_p3_u_regress.nc")
EIMTG_ssp585_p3_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/EIMTG_ssp585_p3_v_regress.nc")
# %%
#   calculate the wind_check and MME
EIMTG_ERA5_wind_mask = ca.wind_check(
    xr.where(EIMTG_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(EIMTG_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(EIMTG_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(EIMTG_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
)

EIMTG_his_wind_mask = ca.wind_check(
    xr.where(EIMTG_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(EIMTG_his_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(EIMTG_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(EIMTG_his_v_pvalue <= 0.05, 1.0, 0.0),
)

EIMTG_ssp585_p3_wind_mask = ca.wind_check(
    xr.where(EIMTG_ssp585_p3_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(EIMTG_ssp585_p3_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(EIMTG_ssp585_p3_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(EIMTG_ssp585_p3_v_pvalue <= 0.05, 1.0, 0.0),
)
#===================================================

EIMTG_his_hgt_slope_ens = EIMTG_his_hgt_slope.mean(dim="models", skipna=True)
EIMTG_his_hgt_rvalue_ens = EIMTG_his_hgt_rvalue.mean(dim="models", skipna=True)

EIMTG_his_hgt_slope_ens_mask = ca.MME_reg_mask(EIMTG_his_hgt_slope_ens, EIMTG_his_hgt_slope.std(dim="models", skipna=True), len(models), True)
EIMTG_his_hgt_rvalue_ens_mask = ca.MME_reg_mask(EIMTG_his_hgt_rvalue_ens, EIMTG_his_hgt_rvalue.std(dim="models", skipna=True), len(models), True)

EIMTG_ssp585_p3_hgt_slope_ens = EIMTG_ssp585_p3_hgt_slope.mean(dim="models", skipna=True)
EIMTG_ssp585_p3_hgt_rvalue_ens = EIMTG_ssp585_p3_hgt_rvalue.mean(dim="models", skipna=True)

EIMTG_ssp585_p3_hgt_slope_ens_mask = ca.MME_reg_mask(EIMTG_ssp585_p3_hgt_slope_ens, EIMTG_ssp585_p3_hgt_slope.std(dim="models", skipna=True), len(models), True)
EIMTG_ssp585_p3_hgt_rvalue_ens_mask = ca.MME_reg_mask(EIMTG_ssp585_p3_hgt_rvalue_ens, EIMTG_ssp585_p3_hgt_rvalue.std(dim="models", skipna=True), len(models), True)
#===================================================
EIMTG_his_u_slope_ens = EIMTG_his_u_slope.mean(dim="models", skipna=True)
EIMTG_his_u_rvalue_ens = EIMTG_his_u_rvalue.mean(dim="models", skipna=True)

EIMTG_his_u_slope_ens_mask = ca.MME_reg_mask(EIMTG_his_u_slope_ens, EIMTG_his_u_slope.std(dim="models", skipna=True), len(models), True)
EIMTG_his_u_rvalue_ens_mask = ca.MME_reg_mask(EIMTG_his_u_rvalue_ens, EIMTG_his_u_rvalue.std(dim="models", skipna=True), len(models), True)

EIMTG_ssp585_p3_u_slope_ens = EIMTG_ssp585_p3_u_slope.mean(dim="models", skipna=True)
EIMTG_ssp585_p3_u_rvalue_ens = EIMTG_ssp585_p3_u_rvalue.mean(dim="models", skipna=True)

EIMTG_ssp585_p3_u_slope_ens_mask = ca.MME_reg_mask(EIMTG_ssp585_p3_u_slope_ens, EIMTG_ssp585_p3_u_slope.std(dim="models", skipna=True), len(models), True)
EIMTG_ssp585_p3_u_rvalue_ens_mask = ca.MME_reg_mask(EIMTG_ssp585_p3_u_rvalue_ens, EIMTG_ssp585_p3_u_rvalue.std(dim="models", skipna=True), len(models), True)
#===================================================
EIMTG_his_v_slope_ens = EIMTG_his_v_slope.mean(dim="models", skipna=True)
EIMTG_his_v_rvalue_ens = EIMTG_his_v_rvalue.mean(dim="models", skipna=True)

EIMTG_his_v_slope_ens_mask = ca.MME_reg_mask(EIMTG_his_v_slope_ens, EIMTG_his_v_slope.std(dim="models", skipna=True), len(models), True)
EIMTG_his_v_rvalue_ens_mask = ca.MME_reg_mask(EIMTG_his_v_rvalue_ens, EIMTG_his_v_rvalue.std(dim="models", skipna=True), len(models), True)

EIMTG_ssp585_p3_v_slope_ens = EIMTG_ssp585_p3_v_slope.mean(dim="models", skipna=True)
EIMTG_ssp585_p3_v_rvalue_ens = EIMTG_ssp585_p3_v_rvalue.mean(dim="models", skipna=True)

EIMTG_ssp585_p3_v_slope_ens_mask = ca.MME_reg_mask(EIMTG_ssp585_p3_v_slope_ens, EIMTG_ssp585_p3_v_slope.std(dim="models", skipna=True), len(models), True)
EIMTG_ssp585_p3_v_rvalue_ens_mask = ca.MME_reg_mask(EIMTG_ssp585_p3_v_rvalue_ens, EIMTG_ssp585_p3_v_rvalue.std(dim="models", skipna=True), len(models), True)
#===================================================
EIMTG_his_wind_slope_ens_mask = ca.wind_check(
    xr.where(EIMTG_his_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(EIMTG_his_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(EIMTG_his_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(EIMTG_his_v_slope_ens_mask > 0.0, 1.0, 0.0),
)
EIMTG_ssp585_p3_wind_slope_ens_mask = ca.wind_check(
    xr.where(EIMTG_ssp585_p3_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(EIMTG_ssp585_p3_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(EIMTG_ssp585_p3_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(EIMTG_ssp585_p3_v_slope_ens_mask > 0.0, 1.0, 0.0),
)
#===================================================
EIMTG_diff_hgt_slope = EIMTG_ssp585_p3_hgt_slope-EIMTG_his_hgt_slope
EIMTG_diff_u_slope = EIMTG_ssp585_p3_u_slope-EIMTG_his_u_slope
EIMTG_diff_v_slope = EIMTG_ssp585_p3_v_slope-EIMTG_his_v_slope
#===================================================
EIMTG_diff_hgt_rvalue = EIMTG_ssp585_p3_hgt_rvalue-EIMTG_his_hgt_rvalue
EIMTG_diff_u_rvalue = EIMTG_ssp585_p3_u_rvalue-EIMTG_his_u_rvalue
EIMTG_diff_v_rvalue = EIMTG_ssp585_p3_v_rvalue-EIMTG_his_v_rvalue
# %%
#   plot the avalue of hgt&u&v regress onto EIMTG in ERA5 and historical
startlevel = np.array([-6.0e-3, -5.0e-3, -3.0e-3])
endlevel = -startlevel
spacinglevel = -startlevel/10.0
scalelevel = [9.5e-5, 6.5e-5, 4.5e-5]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 3
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
        #   EUTT area
        x0 = 60
        y0 = 20
        width = 40
        height = 20.0
        patches(ax, x0 - cl, y0, width, height, proj)
        #   IUTT area
        x0 = 60
        y0 = -10
        width = 40
        height = 20
        patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        EIMTG_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        EIMTG_ERA5_hgt_slope.sel(level=lev), axs[0], n, np.where(EIMTG_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 5.0,
    )
    axs[0].quiver(
        EIMTG_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        EIMTG_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        EIMTG_ERA5_u_slope.sel(level=lev).where(EIMTG_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        EIMTG_ERA5_v_slope.sel(level=lev).where(EIMTG_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=5e-4, label="5e-4", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        EIMTG_his_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        EIMTG_his_hgt_slope_ens.sel(level=lev), axs[1], n, np.where(EIMTG_his_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 5.0,
    )
    axs[1].quiver(
        EIMTG_his_u_slope_ens.sel(level=lev)[::ski, ::ski],
        EIMTG_his_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[1].quiver(
        EIMTG_his_u_slope_ens.sel(level=lev).where(EIMTG_his_wind_slope_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        EIMTG_his_v_slope_ens.sel(level=lev).where(EIMTG_his_wind_slope_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=5e-4, label="5e-4", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+2].contourf(
            EIMTG_his_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            EIMTG_his_hgt_slope.sel(models=mod,level=lev), axs[num_mod+2], n, np.where(EIMTG_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 5.0,
        )
        axs[num_mod+2].quiver(
            EIMTG_his_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            EIMTG_his_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+2].quiver(
            EIMTG_his_u_slope.sel(models=mod,level=lev).where(EIMTG_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            EIMTG_his_v_slope.sel(models=mod,level=lev).where(EIMTG_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+2].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=5e-4, label="5e-4", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+2].format(
            rtitle="1979-2014", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg EIMTG".format(lev))
# %%
#   plot the rvalue of hgt&u&v regress onto EIMTG in ERA5 and historical
startlevel = np.array([-1.0e-0, -1.0e-0, -1.0e-0])
endlevel = -startlevel
spacinglevel = -startlevel/10.0
scalelevel = [0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 29), (7, 4))
    axs = fig.subplots(plot_array, proj=proj)

    #   set the geo_ticks and map projection to the plots
    xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
    yticks = np.arange(-30, 46, 15)  # 设置经度刻度
    # 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
    # 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
    extents = [xticks[0], xticks[-1], yticks[0], 55.0]
    sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
    # ===================================================
    ski = 2
    n = 3
    w, h = 0.12, 0.14
    # ======================================
    for ax in axs:
        rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
        ax.add_patch(rect)
        #   EUTT area
        x0 = 60
        y0 = 20
        width = 40
        height = 20.0
        patches(ax, x0 - cl, y0, width, height, proj)
        #   IUTT area
        x0 = 60
        y0 = -10
        width = 40
        height = 20
        patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        EIMTG_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        EIMTG_ERA5_hgt_rvalue.sel(level=lev), axs[0], n, np.where(EIMTG_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 5.0,
    )
    axs[0].quiver(
        EIMTG_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        EIMTG_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[0].quiver(
        EIMTG_ERA5_u_rvalue.sel(level=lev).where(EIMTG_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        EIMTG_ERA5_v_rvalue.sel(level=lev).where(EIMTG_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=5e-4, label="5e-4", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        EIMTG_his_hgt_rvalue_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        EIMTG_his_hgt_rvalue_ens.sel(level=lev), axs[1], n, np.where(EIMTG_his_hgt_rvalue_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 5.0,
    )
    axs[1].quiver(
        EIMTG_his_u_rvalue_ens.sel(level=lev)[::ski, ::ski],
        EIMTG_his_v_rvalue_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[1].quiver(
        EIMTG_his_u_rvalue_ens.sel(level=lev).where(EIMTG_his_wind_rvalue_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        EIMTG_his_v_rvalue_ens.sel(level=lev).where(EIMTG_his_wind_rvalue_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=5e-4, label="5e-4", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+2].contourf(
            EIMTG_his_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], endlevel[num_lev]+spacinglevel[num_lev]/2, spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            EIMTG_his_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+2], n, np.where(EIMTG_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 5.0,
        )
        axs[num_mod+2].quiver(
            EIMTG_his_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            EIMTG_his_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+2].quiver(
            EIMTG_his_u_rvalue.sel(models=mod,level=lev).where(EIMTG_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            EIMTG_his_v_rvalue.sel(models=mod,level=lev).where(EIMTG_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+2].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=5e-4, label="5e-4", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+2].format(
            rtitle="1979-2014", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg EIMTG".format(lev))
# %%
