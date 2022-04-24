'''
Author: ChenHJ
Date: 2022-04-23 12:49:42
LastEditors: ChenHJ
LastEditTime: 2022-04-24 13:10:33
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
preCRU_JJA = ca.p_time(preCRU, 6, 8, True) / 30.67
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)

hgtERA5_ver_JJA = ca.detrend_dim(hgtERA5_ver_JJA, "time", deg=1, demean=False)
uERA5_ver_JJA = ca.detrend_dim(uERA5_ver_JJA, "time", deg=1, demean=False)
vERA5_ver_JJA = ca.detrend_dim(vERA5_ver_JJA, "time", deg=1, demean=False)
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

EIMTGhis_JJA = ca.detrend_dim(EIMTGhis_JJA, "time", deg=1, demean=False)
EIMTGssp585_JJA = ca.detrend_dim(EIMTGssp585_JJA, "time", deg=1, demean=False)
#   calculate the EUTT-IUTT in ERA5
