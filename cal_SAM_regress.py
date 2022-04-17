'''
Author: ChenHJ
Date: 2022-04-16 23:56:17
LastEditors: ChenHJ
LastEditTime: 2022-04-17 11:34:37
FilePath: /chenhj/0302code/cal_SAM_regress.py
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


def patches(ax, x0, y0, width, height, proj):
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x0, y0), width, height, fc="none", ec="grey7", linewidth=0.8, zorder=1.1, transform=proj, linestyle="--",
    )
    ax.add_patch(rect)
# %%
#   read the ERA5 reanalysis data and calculate the SAM and IWF index
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

fspERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/sp_mon_r144x72_195001-201412.nc"
)
spERA5 = fspERA5["sp"]

fqERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/q_mon_r144x72_195001-201412.nc")
qERA5 = fqERA5["q"]

# %%
#   read the precipitation data
fpreCRU = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]

# GPCP data just have 1979-2014 year
fpreGPCP = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]

# %%
hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True).loc[:, 100.0:, :, :]
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True).loc[:, 100.0:, :, :]
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True).loc[:, 100.0:, :, :]
qERA5_ver_JJA = ca.p_time(qERA5, 6, 9, True).loc[:, 100.0:, :, :]
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)
preCRU_JJA = ca.p_time(preCRU, 6, 8, True) / 30.67
preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)

hgtERA5_ver_JJA = ca.detrend_dim(hgtERA5_ver_JJA, "time", deg=1, demean=False)
uERA5_ver_JJA = ca.detrend_dim(uERA5_ver_JJA, "time", deg=1, demean=False)
vERA5_ver_JJA = ca.detrend_dim(vERA5_ver_JJA, "time", deg=1, demean=False)
qERA5_ver_JJA = ca.detrend_dim(qERA5_ver_JJA, "time", deg=1, demean=False)
spERA5_JJA = ca.detrend_dim(spERA5_JJA, "time", deg=1, demean=False)
preCRU_JJA = ca.detrend_dim(preCRU_JJA, "time", deg=1, demean=False)
preGPCP_JJA = ca.detrend_dim(preGPCP_JJA, "time", deg=1, demean=False)

# %%
#   calculate the SAM index in ERA5
ERA5_SAM_index = ca.SAM(vERA5_ver_JJA)
ERA5_SAM_index = ca.detrend_dim(ERA5_SAM_index, "time", deg=1, demean=False)

# %%
# read the SAM index and precipitation in historical run and ssp585
fhis_SAM_index = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_SAM_index_1950-2014.nc")
his_SAM_index = fhis_SAM_index["SAM"]

fssp585_SAM_index = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_SAM_index_2015-2099.nc")
ssp585_SAM_index = fssp585_SAM_index["SAM"]

fprehis_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pr_historical_r144x72_195001-201412.nc")
prehis_JJA = fprehis_JJA["pr"]

fpressp585_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pr_ssp585_r144x72_201501-209912.nc")
pressp585_JJA = fpressp585_JJA["pr"]
# %%
#   pick up the precipitation in India area
lat = preCRU_JJA.coords["lat"]
lon = preCRU_JJA.coords["lon"]
lat_India_range = lat[(lat >= 8.0) & (lat <= 28.0)]
lon_India_range = lon[(lon >= 70.0) & (lon <= 86.0)]
preCRU_India_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

lat = preGPCP_JJA.coords["lat"]
lon = preGPCP_JJA.coords["lon"]
lat_India_range = lat[(lat >= 8.0) & (lat <= 28.0)]
lon_India_range = lon[(lon >= 70.0) & (lon <= 86.0)]
preGPCP_India_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

lat = prehis_JJA.coords["lat"]
lon = prehis_JJA.coords["lon"]
lat_India_range = lat[(lat >= 8.0) & (lat <= 28.0)]
lon_India_range = lon[(lon >= 70.0) & (lon <= 86.0)]
prehis_India_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

lat = pressp585_JJA.coords["lat"]
lon = pressp585_JJA.coords["lon"]
lat_India_range = lat[(lat >= 8.0) & (lat <= 28.0)]
lon_India_range = lon[(lon >= 70.0) & (lon <= 86.0)]
pressp585_India_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

del(lat, lon)

#   calculate the correlation coefficients between SAM and IndR
IndRGPCP_ERA5_SAM_regress = stats.linregress(preGPCP_India_JJA, ERA5_SAM_index.sel(time=ERA5_SAM_index.time.dt.year>=1979))
IndR_ERA5_SAM_regress = stats.linregress(preCRU_India_JJA, ERA5_SAM_index)
IndR_his_SAM_regress = ca.dim_linregress(prehis_India_JJA, his_SAM_index)
IndR_ssp585_SAM_regress = ca.dim_linregress(pressp585_India_JJA, ssp585_SAM_index)
# %%
