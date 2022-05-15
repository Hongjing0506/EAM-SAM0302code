'''
Author: ChenHJ
Date: 2022-05-13 22:29:49
LastEditors: ChenHJ
LastEditTime: 2022-05-15 21:49:18
FilePath: /chenhj/0302code/cal_NC_regress.py
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
import metpy.constants as constants
import geocat.comp
from windspharm.xarray import VectorWind
import skill_metrics as sm
from brokenaxes import brokenaxes

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
#   read the data in CRU/GPCP/ERA5/historical/ssp585
# fpreCRU = xr.open_dataset(
#     "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
# )
# preCRU = fpreCRU["pre"]
# preCRU_JJA = ca.p_time(preCRU, 6, 8, True)/30.67
# preCRU_JJA = ca.detrend_dim(preCRU_JJA, "time", deg=1, demean=False)
# preCRU_JJA.attrs["units"] = "mm/day"

fpreGPCP = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]
preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)
preGPCP_JJA = ca.detrend_dim(preGPCP_JJA, "time", deg=1, demean=False)

preAIR = xr.open_dataarray("/home/ys17-23/Extension/All_India_Rainfall_index/AIR_mmperday.nc")
preAIR_JJA = ca.p_time(preAIR, 6, 8, True)
preAIR_JJA = preAIR_JJA.sel(time=(preAIR_JJA.time.dt.year>=1979) & (preAIR_JJA.time.dt.year <=2014))
preAIR_JJA = ca.detrend_dim(preAIR_JJA, "time", deg=1, demean=False)

fprehis = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pr_historical_r144x72_195001-201412.nc")
prehis_JJA = fprehis["pr"].sel(time=fprehis["time"].dt.year>=1979)
prehis_JJA.attrs["units"] = "mm/day"
prehis_JJA.attrs["standard_name"] = "precipitation"
prehis_JJA = ca.detrend_dim(prehis_JJA, "time", deg=1, demean=False)

fpressp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pr_ssp585_r144x72_201501-209912.nc")
pressp585_JJA = fpressp585["pr"]
pressp585_JJA.attrs["units"] = "mm/day"
pressp585_JJA.attrs["standard_name"] = "precipitation"
pressp585_p3_JJA = pressp585_JJA.sel(time=pressp585_JJA.time.dt.year>=2064)
pressp585_p3_JJA = ca.detrend_dim(pressp585_p3_JJA, "time", deg=1, demean=False)

fhgtERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/hgt_mon_r144x72_195001-201412.nc")
hgtERA5 = fhgtERA5["z"].sel(time=fhgtERA5["time"].dt.year>=1979)

fuERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/uwind_mon_r144x72_195001-201412.nc"
)
uERA5 = fuERA5["u"].sel(time=fuERA5["time"].dt.year>=1979)

fvERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/vwind_mon_r144x72_195001-201412.nc"
)
vERA5 = fvERA5["v"].sel(time=fvERA5["time"].dt.year>=1979)

fspERA5 = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/sp_mon_r144x72_195001-201412.nc"
)
spERA5 = fspERA5["sp"].sel(time=fspERA5["time"].dt.year>=1979)

fqERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/q_mon_r144x72_195001-201412.nc")
qERA5 = fqERA5["q"].sel(time=fqERA5["time"].dt.year>=1979)

fwERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/omega_mon_r144x72_195001-201412.nc")
wERA5 = fwERA5["w"].sel(time=fwERA5["time"].dt.year>=1979)

hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True)
hgtERA5_ver_JJA = hgtERA5_ver_JJA-hgtERA5_ver_JJA.mean(dim="lon", skipna=True)
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True)
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True)
qERA5_ver_JJA = ca.p_time(qERA5, 6, 9, True)
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)
wERA5_JJA = ca.p_time(wERA5, 6, 8, True)

hgtERA5_ver_JJA = ca.detrend_dim(hgtERA5_ver_JJA, "time", deg=1, demean=False)
uERA5_ver_JJA = ca.detrend_dim(uERA5_ver_JJA, "time", deg=1, demean=False)
vERA5_ver_JJA = ca.detrend_dim(vERA5_ver_JJA, "time", deg=1, demean=False)
qERA5_ver_JJA = ca.detrend_dim(qERA5_ver_JJA, "time", deg=1, demean=False)
spERA5_JJA = ca.detrend_dim(spERA5_JJA, "time", deg=1, demean=False)
wERA5_JJA = ca.detrend_dim(wERA5_JJA, "time", deg=1, demean=False)

fhgthis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/zg_historical_r144x72_195001-201412.nc")
hgthis_ver_JJA = fhgthis_ver_JJA["zg"].sel(time=fhgthis_ver_JJA["time"].dt.year>=1979)
hgthis_ver_JJA = hgthis_ver_JJA - hgthis_ver_JJA.mean(dim="lon", skipna=True)
hgthis_ver_JJA = ca.detrend_dim(hgthis_ver_JJA, "time", deg=1, demean=False)

fuhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ua_historical_r144x72_195001-201412.nc")
uhis_ver_JJA = fuhis_ver_JJA["ua"].sel(time=fuhis_ver_JJA["time"].dt.year>=1979)
uhis_ver_JJA = ca.detrend_dim(uhis_ver_JJA, "time", deg=1, demean=False)

fvhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/va_historical_r144x72_195001-201412.nc")
vhis_ver_JJA = fvhis_ver_JJA["va"].sel(time=fvhis_ver_JJA["time"].dt.year>=1979)
vhis_ver_JJA = ca.detrend_dim(vhis_ver_JJA, "time", deg=1, demean=False)

fwhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/wap_historical_r144x72_195001-201412.nc") 
whis_ver_JJA = fwhis_ver_JJA["wap"].sel(time=fwhis_ver_JJA["time"].dt.year>=1979)
whis_ver_JJA = ca.detrend_dim(whis_ver_JJA, "time", deg=1, demean=False)

fhgtssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/zg_ssp585_r144x72_201501-209912.nc")
hgtssp585_ver_JJA = fhgtssp585_ver_JJA["zg"]
hgtssp585_p3_ver_JJA = hgtssp585_ver_JJA.sel(time=hgtssp585_ver_JJA.time.dt.year>=2064)


hgtssp585_ver_JJA = hgtssp585_ver_JJA - hgtssp585_ver_JJA.mean(dim="lon", skipna=True)
hgtssp585_p3_ver_JJA = hgtssp585_p3_ver_JJA - hgtssp585_p3_ver_JJA.mean(dim="lon", skipna=True)
hgtssp585_p3_ver_JJA = ca.detrend_dim(hgtssp585_p3_ver_JJA, "time", deg=1, demean=False)


fussp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ua_ssp585_r144x72_201501-209912.nc")
ussp585_ver_JJA = fussp585_ver_JJA["ua"]
ussp585_p3_ver_JJA = ussp585_ver_JJA.sel(time=ussp585_ver_JJA.time.dt.year>=2064)
ussp585_p3_ver_JJA = ca.detrend_dim(ussp585_p3_ver_JJA, "time", deg=1, demean=False)

fvssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/va_ssp585_r144x72_201501-209912.nc")
vssp585_ver_JJA = fvssp585_ver_JJA["va"]
vssp585_p3_ver_JJA = vssp585_ver_JJA.sel(time=vssp585_ver_JJA.time.dt.year>=2064)
vssp585_p3_ver_JJA = ca.detrend_dim(vssp585_p3_ver_JJA, "time", deg=1, demean=False)

fwssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/wap_ssp585_r144x72_201501-209912.nc")
wssp585_ver_JJA = fwssp585_ver_JJA["wap"]
wssp585_p3_ver_JJA = wssp585_ver_JJA.sel(time=wssp585_ver_JJA.time.dt.year>=2064)
wssp585_p3_ver_JJA = ca.detrend_dim(wssp585_p3_ver_JJA, "time", deg=1, demean=False)

#read the temperature data in ERA5/historical/ssp585
ftERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/temp_mon_r144x72_195001-201412.nc")
tERA5 = ftERA5["t"].sel(time=ftERA5["time"].dt.year>=1979)
tERA5_ver_JJA = ca.p_time(tERA5, 6, 8, True)
tERA5_ver_JJA = ca.detrend_dim(tERA5_ver_JJA, "time", deg=1, demean=False)

fthis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ta_historical_r144x72_195001-201412.nc")
this_ver_JJA = fthis_ver_JJA["ta"].sel(time=fthis_ver_JJA["time"].dt.year>=1979)
this_ver_JJA = ca.detrend_dim(this_ver_JJA, "time", deg=1, demean=False)

ftssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ta_ssp585_r144x72_201501-209912.nc")
tssp585_ver_JJA = ftssp585_ver_JJA["ta"]
tssp585_p3_ver_JJA = tssp585_ver_JJA.sel(time=tssp585_ver_JJA.time.dt.year>=2064)
tssp585_p3_ver_JJA = ca.detrend_dim(tssp585_p3_ver_JJA, "time", deg=1, demean=False)

#   read the his_dpg and ssp585_dpg
# his_dsdpg = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_dsdpg500-200.nc")
# ssp585_dsdpg = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_dsdpg500-200.nc")

# #   calculate the utt in historical and ssp585
# utthis_JJA = (this_ver_JJA.loc[:,:,500.0:200.0,:,:]*his_dsdpg.data).sum(dim="level",skipna=True)
# uttssp585_JJA = (tssp585_ver_JJA.loc[:,:,500.0:200.0,:,:]*ssp585_dsdpg.data).sum(dim="level",skipna=True)
# utthis_JJA = ca.detrend_dim(utthis_JJA, "time", deg=1, demean=False)
# uttssp585_JJA = ca.detrend_dim(uttssp585_JJA, "time", deg=1, demean=False)
# utthis_JJA.name="utt"
# uttssp585_JJA.name="utt"

#   deal with the time index for CRU and GPCP data
# preCRU_JJA.coords["time"] = prehis_JJA.coords["time"]
preGPCP_JJA.coords["time"] = prehis_JJA.sel(time=prehis_JJA.time.dt.year>=1979).coords["time"]
preAIR_JJA.coords["time"] = prehis_JJA.coords["time"]
# %%
models = uhis_ver_JJA.coords["models"]
models_array = models.data
# %%
# #   calculate the ERA5 upper level troposphere temperature between 500hPa to 200hPa
# ptop = 1 * 200
# g = 9.8
# ERA5_dslevel = uERA5_ver_JJA.coords["level"].loc[200.0:500.0] * 100.0
# ERA5_dslevel.attrs["units"] = "Pa"
# ERA5_dsdp = geocat.comp.dpres_plevel(ERA5_dslevel, spERA5_JJA, ptop)
# ERA5_dsdpg = ERA5_dsdp / g
# ERA5_dsdpg.attrs["units"] = "kg/m2"
# ERA5_dsdpg.name = "dsdpg"

# uttERA5_JJA = (tERA5_ver_JJA.loc[:,200.0:500.0,:,:] * ERA5_dsdpg.data).sum(dim="level", skipna=True)
# uttERA5_JJA = ca.detrend_dim(uttERA5_JJA, "time", deg=1, demean=False)
# uttERA5_JJA.name = "utt"
# %%
#   pick up the area data
#   calculate the vorticity in ERA5, historical and ssp585
vorERA5_ver_JJA = mpcalc.vorticity(uERA5_ver_JJA.sel(level=200.0), vERA5_ver_JJA.sel(level=200.0))
vorERA5_ver_JJA = vorERA5_ver_JJA.metpy.dequantify()

vorhis_ver_JJA = mpcalc.vorticity(uhis_ver_JJA.sel(level=200.0), vhis_ver_JJA.sel(level=200.0))
vorhis_ver_JJA = vorhis_ver_JJA.metpy.dequantify()

vorssp585_ver_JJA = mpcalc.vorticity(ussp585_ver_JJA.sel(level=200.0), vssp585_ver_JJA.sel(level=200.0))
vorssp585_ver_JJA = vorssp585_ver_JJA.metpy.dequantify()

vorssp585_p3_ver_JJA = mpcalc.vorticity(ussp585_p3_ver_JJA.sel(level=200.0), vssp585_p3_ver_JJA.sel(level=200.0))
vorssp585_p3_ver_JJA = vorssp585_p3_ver_JJA.metpy.dequantify()

#   calculate the precipitation in India
lat = preGPCP_JJA.coords["lat"]
lon = preGPCP_JJA.coords["lon"]

India_N = 32.5
# India_N = 30.0
India_S = 8.0
India_W = 70.0
India_E = 86.0
lat_India_range = lat[(lat >= India_S) & (lat <= India_N)]
lon_India_range = lon[(lon >= India_W) & (lon <= India_E)]

# preCRU_India_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
preGPCP_India_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
prehis_India_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
pressp585_India_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
pressp585_p3_India_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Northern China
NC_N = 40.0
NC_S = 32.5
NC_W = 105.0
NC_E = 120.0
lat_NC_range = lat[(lat >= NC_S) & (lat <= NC_N)]
lon_NC_range = lon[(lon >= NC_W) & (lon <= NC_E)]
# lat_NC_range = lat[(lat>=27.5) & (lat<=37.5)]
# lon_NC_range = lon[(lon>=105.0) & (lon<=125.0)]
# preCRU_NC_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)
preGPCP_NC_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)
prehis_NC_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)
pressp585_NC_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)
pressp585_p3_NC_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Southern China
lat_SC_range = lat[(lat>=20.0) & (lat<=27.5)]
lon_SC_range = lon[(lon>=105.0) & (lon<=125.0)]
# preCRU_SC_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
preGPCP_SC_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
prehis_SC_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
pressp585_SC_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
pressp585_p3_SC_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Korean Peninsula
KP_N = 37.5
KP_S = 32.5
KP_W = 124.0
KP_E = 132.0
lat_KP_range = lat[(lat >= KP_S) & (lat <= KP_N)]
lon_KP_range = lon[(lon >= KP_W) & (lon <= KP_E)]
# lat_KP_range = lat[(lat>=27.5) & (lat<=37.5)]
# lon_KP_range = lon[(lon>=105.0) & (lon<=125.0)]
# preCRU_KP_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)
preGPCP_KP_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)
prehis_KP_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)
pressp585_KP_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)
pressp585_p3_KP_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Korean Peninsula-Southern Japan
SJ_N = 31.0
SJ_S = 24.0
SJ_W = 124.0
SJ_E = 136.0
lat_SJ_range = lat[(lat >= SJ_S) & (lat <= SJ_N)]
lon_SJ_range = lon[(lon >= SJ_W) & (lon <= SJ_E)]
# lat_SJ_range = lat[(lat>=27.5) & (lat<=37.5)]
# lon_SJ_range = lon[(lon>=105.0) & (lon<=125.0)]
# preCRU_SJ_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)
preGPCP_SJ_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)
prehis_SJ_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)
pressp585_SJ_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)
pressp585_p3_SJ_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)

#   calculate the 200hPa u-wind over the East Asia
lat_EA_range = lat[(lat>=30.0) & (lat<=40.0)]
lon_EA_range = lon[(lon>=100.0) & (lon<=120.0)]

uERA5_EA_JJA = ca.cal_lat_weighted_mean(uERA5_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)
uhis_EA_JJA = ca.cal_lat_weighted_mean(uhis_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)
ussp585_EA_JJA = ca.cal_lat_weighted_mean(ussp585_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)
ussp585_p3_EA_JJA = ca.cal_lat_weighted_mean(ussp585_p3_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)

#   calculate the 200hPa vorticity over the East Asia
lat_EAhigh_range = lat[(lat>=22.5) & (lat<=50.0)]
lon_EAhigh_range = lon[(lon>=115.0) & (lon<=140.0)]
uERA5_EAhigh_JJA = uERA5_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
uhis_EAhigh_JJA = uhis_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
ussp585_EAhigh_JJA = ussp585_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
ussp585_p3_EAhigh_JJA = ussp585_p3_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)

vERA5_EAhigh_JJA = vERA5_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
vhis_EAhigh_JJA = vhis_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
vssp585_EAhigh_JJA = vssp585_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
vssp585_p3_EAhigh_JJA = vssp585_p3_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)

vorERA5_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uERA5_EAhigh_JJA, vERA5_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorhis_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uhis_EAhigh_JJA, vhis_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_EAhigh_JJA, vssp585_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_p3_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_p3_EAhigh_JJA, vssp585_p3_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()

# #   calculate the longitude mean over 100.0° to 125°E
# lon_EA_range = lon[(lon>=100.0)&(lon<=125.0)]

# uERA5_EA_lm_JJA = uERA5_ver_JJA.loc[:,10.0:,0.:,:].sel(lon=lon_EA_range).mean(dim="lon",skipna=True)
# uhis_EA_lm_JJA = uhis_ver_JJA.loc[:,:,:10.0,0.:,:].sel(lon=lon_EA_range).mean(dim="lon",skipna=True)
# ussp585_EA_lm_JJA = ussp585_ver_JJA.loc[:,:,:10.0,0.:,:].sel(lon=lon_EA_range).mean(dim="lon",skipna=True)

# tERA5_EA_lm_JJA = tERA5_ver_JJA.loc[:,10.0:,0.:,:].sel(lon=lon_EA_range).mean(dim="lon",skipna=True)
# this_EA_lm_JJA = this_ver_JJA.loc[:,:,:10.0,0.:,:].sel(lon=lon_EA_range).mean(dim="lon",skipna=True)
# tssp585_EA_lm_JJA = tssp585_ver_JJA.loc[:,:,:10.0,0.:,:].sel(lon=lon_EA_range).mean(dim="lon",skipna=True)

#   calculate the MTG related to the 200hPa u wind over East Asia
# lat_area1_range = lat[(lat >= 15.0) & (lat <= 30.0)]
# lon_area1_range = lon[(lon >= 100.0) & (lon <= 125.0)]

# lat_area2_range = lat[(lat >= 33.75) & (lat <= 45.0)]
# lon_area2_range = lon[(lon >= 100.0) & (lon <= 125.0)]

# EAU_MTGERA5_JJA = ca.cal_lat_weighted_mean(uttERA5_JJA.sel(lat=lat_area2_range,lon=lon_area2_range)).mean(dim="lon",skipna=True)-ca.cal_lat_weighted_mean(uttERA5_JJA.sel(lat=lat_area1_range,lon=lon_area1_range)).mean(dim="lon",skipna=True)
# EAU_MTGhis_JJA = ca.cal_lat_weighted_mean(utthis_JJA.sel(lat=lat_area2_range,lon=lon_area2_range)).mean(dim="lon",skipna=True)-ca.cal_lat_weighted_mean(utthis_JJA.sel(lat=lat_area1_range,lon=lon_area1_range)).mean(dim="lon",skipna=True)
# EAU_MTGssp585_JJA = ca.cal_lat_weighted_mean(uttssp585_JJA.sel(lat=lat_area2_range,lon=lon_area2_range)).mean(dim="lon",skipna=True)-ca.cal_lat_weighted_mean(uttssp585_JJA.sel(lat=lat_area1_range,lon=lon_area1_range)).mean(dim="lon",skipna=True)

#   calculate the vorticity over the West Asia in 200hPa
lat_WAhigh_range = lat[(lat>=25.0) & (lat<=45.0)]
lon_WAhigh_range = lon[(lon>=55.0) & (lon<=75.0)]
uERA5_WAhigh_JJA = uERA5_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
uhis_WAhigh_JJA = uhis_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
ussp585_WAhigh_JJA = ussp585_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
ussp585_p3_WAhigh_JJA = ussp585_p3_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)

vERA5_WAhigh_JJA = vERA5_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
vhis_WAhigh_JJA = vhis_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
vssp585_WAhigh_JJA = vssp585_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
vssp585_p3_WAhigh_JJA = vssp585_p3_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)

vorERA5_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uERA5_WAhigh_JJA, vERA5_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorhis_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uhis_WAhigh_JJA, vhis_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_WAhigh_JJA, vssp585_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_p3_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_p3_WAhigh_JJA, vssp585_p3_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
# %%
#   calculate the detrend-data for the picked-up area data
vorERA5_ver_JJA = ca.detrend_dim(vorERA5_ver_JJA, "time", deg=1, demean=False)
vorhis_ver_JJA = ca.detrend_dim(vorhis_ver_JJA, "time", deg=1, demean=False)
vorssp585_ver_JJA = ca.detrend_dim(vorssp585_ver_JJA, "time", deg=1, demean=False)
vorssp585_p3_ver_JJA = ca.detrend_dim(vorssp585_p3_ver_JJA, "time", deg=1, demean=False)

# preCRU_India_JJA = ca.detrend_dim(preCRU_India_JJA, "time", deg=1, demean=False)
preGPCP_India_JJA = ca.detrend_dim(preGPCP_India_JJA, "time", deg=1, demean=False)
prehis_India_JJA = ca.detrend_dim(prehis_India_JJA, "time", deg=1, demean=False)
pressp585_India_JJA = ca.detrend_dim(pressp585_India_JJA, "time", deg=1, demean=False)
pressp585_p3_India_JJA = ca.detrend_dim(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), "time", deg=1, demean=False)

# preCRU_NC_JJA = ca.detrend_dim(preCRU_NC_JJA, "time", deg=1, demean=False)
preGPCP_NC_JJA = ca.detrend_dim(preGPCP_NC_JJA, "time", deg=1, demean=False)
prehis_NC_JJA = ca.detrend_dim(prehis_NC_JJA, "time", deg=1, demean=False)
pressp585_NC_JJA = ca.detrend_dim(pressp585_NC_JJA, "time", deg=1, demean=False)
pressp585_p3_NC_JJA = ca.detrend_dim(pressp585_p3_NC_JJA, "time", deg=1, demean=False)

preGPCP_KP_JJA = ca.detrend_dim(preGPCP_KP_JJA, "time", deg=1, demean=False)
prehis_KP_JJA = ca.detrend_dim(prehis_KP_JJA, "time", deg=1, demean=False)
pressp585_KP_JJA = ca.detrend_dim(pressp585_KP_JJA, "time", deg=1, demean=False)
pressp585_p3_KP_JJA = ca.detrend_dim(pressp585_p3_KP_JJA, "time", deg=1, demean=False)


preGPCP_SJ_JJA = ca.detrend_dim(preGPCP_SJ_JJA, "time", deg=1, demean=False)
prehis_SJ_JJA = ca.detrend_dim(prehis_SJ_JJA, "time", deg=1, demean=False)
pressp585_SJ_JJA = ca.detrend_dim(pressp585_SJ_JJA, "time", deg=1, demean=False)
pressp585_p3_SJ_JJA = ca.detrend_dim(pressp585_p3_SJ_JJA, "time", deg=1, demean=False)


preGPCP_SC_JJA = ca.detrend_dim(preGPCP_SC_JJA, "time", deg=1, demean=False)
prehis_SC_JJA = ca.detrend_dim(prehis_SC_JJA, "time", deg=1, demean=False)
pressp585_SC_JJA = ca.detrend_dim(pressp585_SC_JJA, "time", deg=1, demean=False)
pressp585_p3_SC_JJA = ca.detrend_dim(pressp585_p3_SC_JJA, "time", deg=1, demean=False)

uERA5_EA_JJA = ca.detrend_dim(uERA5_EA_JJA, "time", deg=1, demean=False)
uhis_EA_JJA = ca.detrend_dim(uhis_EA_JJA, "time", deg=1, demean=False)
ussp585_EA_JJA = ca.detrend_dim(ussp585_EA_JJA, "time", deg=1, demean=False)
ussp585_p3_EA_JJA = ca.detrend_dim(ussp585_p3_EA_JJA, "time", deg=1, demean=False)

vorERA5_EAhigh_JJA = ca.detrend_dim(vorERA5_EAhigh_JJA, "time", deg=1, demean=False)
vorhis_EAhigh_JJA = ca.detrend_dim(vorhis_EAhigh_JJA, "time", deg=1, demean=False)
vorssp585_EAhigh_JJA = ca.detrend_dim(vorssp585_EAhigh_JJA, "time", deg=1, demean=False)
vorssp585_p3_EAhigh_JJA = ca.detrend_dim(vorssp585_p3_EAhigh_JJA, "time", deg=1, demean=False)

# uERA5_EA_lm_JJA = ca.detrend_dim(uERA5_EA_lm_JJA, "time", deg=1, demean=False)
# uhis_EA_lm_JJA = ca.detrend_dim(uhis_EA_lm_JJA, "time", deg=1, demean=False)
# ussp585_EA_lm_JJA = ca.detrend_dim(ussp585_EA_lm_JJA, "time", deg=1, demean=False)

# tERA5_EA_lm_JJA = ca.detrend_dim(tERA5_EA_lm_JJA, "time", deg=1, demean=False)
# this_EA_lm_JJA = ca.detrend_dim(this_EA_lm_JJA, "time", deg=1, demean=False)
# tssp585_EA_lm_JJA = ca.detrend_dim(tssp585_EA_lm_JJA, "time", deg=1, demean=False)

# EAU_MTGhis_JJA = ca.detrend_dim(EAU_MTGhis_JJA, "time", deg=1, demean=False)
# EAU_MTGssp585_JJA = ca.detrend_dim(EAU_MTGssp585_JJA, "time", deg=1, demean=False)
# EAU_MTGERA5_JJA = ca.detrend_dim(EAU_MTGERA5_JJA, "time", deg=1, demean=False)

vorERA5_WAhigh_JJA = ca.detrend_dim(vorERA5_WAhigh_JJA, "time", deg=1, demean=False)
vorhis_WAhigh_JJA = ca.detrend_dim(vorhis_WAhigh_JJA, "time", deg=1, demean=False)
vorssp585_WAhigh_JJA = ca.detrend_dim(vorssp585_WAhigh_JJA, "time", deg=1, demean=False)
vorssp585_p3_WAhigh_JJA = ca.detrend_dim(vorssp585_p3_WAhigh_JJA, "time", deg=1, demean=False)
# %%
#   calculate the hgt/u/v regression onto NCR in ERA5, historical, ssp585, ssp585_p3
preGPCP_NC_JJA.coords["time"] = hgtERA5_ver_JJA.sel(time=hgtERA5_ver_JJA.time.dt.year>=1979).coords["time"]


(
    NCRGPCP_ERA5_hgt_slope,
    NCRGPCP_ERA5_hgt_intercept,
    NCRGPCP_ERA5_hgt_rvalue,
    NCRGPCP_ERA5_hgt_pvalue,
    NCRGPCP_ERA5_hgt_hypothesis,
) = ca.dim_linregress(preGPCP_NC_JJA, hgtERA5_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    NCRGPCP_ERA5_u_slope,
    NCRGPCP_ERA5_u_intercept,
    NCRGPCP_ERA5_u_rvalue,
    NCRGPCP_ERA5_u_pvalue,
    NCRGPCP_ERA5_u_hypothesis,
) = ca.dim_linregress(preGPCP_NC_JJA, uERA5_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    NCRGPCP_ERA5_v_slope,
    NCRGPCP_ERA5_v_intercept,
    NCRGPCP_ERA5_v_rvalue,
    NCRGPCP_ERA5_v_pvalue,
    NCRGPCP_ERA5_v_hypothesis,
) = ca.dim_linregress(preGPCP_NC_JJA, vERA5_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    NCR_his_hgt_slope,
    NCR_his_hgt_intercept,
    NCR_his_hgt_rvalue,
    NCR_his_hgt_pvalue,
    NCR_his_hgt_hypothesis,
) = ca.dim_linregress(prehis_NC_JJA, hgthis_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    NCR_his_u_slope,
    NCR_his_u_intercept,
    NCR_his_u_rvalue,
    NCR_his_u_pvalue,
    NCR_his_u_hypothesis,
) = ca.dim_linregress(prehis_NC_JJA, uhis_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    NCR_his_v_slope,
    NCR_his_v_intercept,
    NCR_his_v_rvalue,
    NCR_his_v_pvalue,
    NCR_his_v_hypothesis,
) = ca.dim_linregress(prehis_NC_JJA, vhis_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    NCR_ssp585_p3_hgt_slope,
    NCR_ssp585_p3_hgt_intercept,
    NCR_ssp585_p3_hgt_rvalue,
    NCR_ssp585_p3_hgt_pvalue,
    NCR_ssp585_p3_hgt_hypothesis,
) = ca.dim_linregress(pressp585_p3_NC_JJA, hgtssp585_p3_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    NCR_ssp585_p3_u_slope,
    NCR_ssp585_p3_u_intercept,
    NCR_ssp585_p3_u_rvalue,
    NCR_ssp585_p3_u_pvalue,
    NCR_ssp585_p3_u_hypothesis,
) = ca.dim_linregress(pressp585_p3_NC_JJA, ussp585_p3_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

(
    NCR_ssp585_p3_v_slope,
    NCR_ssp585_p3_v_intercept,
    NCR_ssp585_p3_v_rvalue,
    NCR_ssp585_p3_v_pvalue,
    NCR_ssp585_p3_v_hypothesis,
) = ca.dim_linregress(pressp585_p3_NC_JJA, vssp585_p3_ver_JJA.sel(level=[200.0, 500.0, 850.0]))


# %%
#   save the regression result
level=NCR_his_hgt_slope.coords["level"]
lat=NCR_his_hgt_slope.coords["lat"]
lon=NCR_his_hgt_slope.coords["lon"]

NCRGPCP_ERA5_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], NCRGPCP_ERA5_hgt_slope.data),
        intercept=(["level", "lat", "lon"], NCRGPCP_ERA5_hgt_intercept.data),
        rvalue=(["level", "lat", "lon"], NCRGPCP_ERA5_hgt_rvalue.data),
        pvalue=(["level", "lat", "lon"], NCRGPCP_ERA5_hgt_pvalue.data),
        hypothesis=(["level", "lat", "lon"], NCRGPCP_ERA5_hgt_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of ERA5 regress onto 1979-2014 GPCP NCR"),
)

NCRGPCP_ERA5_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], NCRGPCP_ERA5_u_slope.data),
        intercept=(["level", "lat", "lon"], NCRGPCP_ERA5_u_intercept.data),
        rvalue=(["level", "lat", "lon"], NCRGPCP_ERA5_u_rvalue.data),
        pvalue=(["level", "lat", "lon"], NCRGPCP_ERA5_u_pvalue.data),
        hypothesis=(["level", "lat", "lon"], NCRGPCP_ERA5_u_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of ERA5 regress onto 1979-2014 GPCP NCR"),
)

NCRGPCP_ERA5_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], NCRGPCP_ERA5_v_slope.data),
        intercept=(["level", "lat", "lon"], NCRGPCP_ERA5_v_intercept.data),
        rvalue=(["level", "lat", "lon"], NCRGPCP_ERA5_v_rvalue.data),
        pvalue=(["level", "lat", "lon"], NCRGPCP_ERA5_v_pvalue.data),
        hypothesis=(["level", "lat", "lon"], NCRGPCP_ERA5_v_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of ERA5 regress onto 1979-2014 GPCP NCR"),
)

NCR_his_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], NCR_his_hgt_slope.data),
        intercept=(["models", "level", "lat", "lon"], NCR_his_hgt_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], NCR_his_hgt_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], NCR_his_hgt_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], NCR_his_hgt_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of historical regress onto 1979-2014 NCR"),
)

NCR_his_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], NCR_his_u_slope.data),
        intercept=(["models", "level", "lat", "lon"], NCR_his_u_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], NCR_his_u_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], NCR_his_u_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], NCR_his_u_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of historical regress onto 1979-2014 NCR"),
)

NCR_his_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], NCR_his_v_slope.data),
        intercept=(["models", "level", "lat", "lon"], NCR_his_v_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], NCR_his_v_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], NCR_his_v_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], NCR_his_v_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of historical regress onto 1979-2014 NCR"),
)

NCR_ssp585_p3_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], NCR_ssp585_p3_hgt_slope.data),
        intercept=(["models", "level", "lat", "lon"], NCR_ssp585_p3_hgt_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], NCR_ssp585_p3_hgt_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], NCR_ssp585_p3_hgt_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], NCR_ssp585_p3_hgt_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of ssp585_p3 regress onto 2064-2099 NCR"),
)

NCR_ssp585_p3_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], NCR_ssp585_p3_u_slope.data),
        intercept=(["models", "level", "lat", "lon"], NCR_ssp585_p3_u_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], NCR_ssp585_p3_u_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], NCR_ssp585_p3_u_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], NCR_ssp585_p3_u_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of ssp585_p3 regress onto 2064-2099 NCR"),
)

NCR_ssp585_p3_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], NCR_ssp585_p3_v_slope.data),
        intercept=(["models", "level", "lat", "lon"], NCR_ssp585_p3_v_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], NCR_ssp585_p3_v_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], NCR_ssp585_p3_v_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], NCR_ssp585_p3_v_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of ssp585_p3 regress onto 2064-2099 NCR"),
)

NCRGPCP_ERA5_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCRGPCP_ERA5_hgt_regress.nc")
NCRGPCP_ERA5_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCRGPCP_ERA5_u_regress.nc")
NCRGPCP_ERA5_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCRGPCP_ERA5_v_regress.nc")

NCR_his_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCR_his_hgt_regress.nc")
NCR_his_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCR_his_u_regress.nc")
NCR_his_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCR_his_v_regress.nc")

NCR_ssp585_p3_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/NCR_ssp585_p3_hgt_regress.nc")
NCR_ssp585_p3_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/NCR_ssp585_p3_u_regress.nc")
NCR_ssp585_p3_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/NCR_ssp585_p3_v_regress.nc")

# %%
#   read the data
NCRGPCP_ERA5_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCRGPCP_ERA5_hgt_regress.nc")
NCRGPCP_ERA5_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCRGPCP_ERA5_u_regress.nc")
NCRGPCP_ERA5_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCRGPCP_ERA5_v_regress.nc")

NCR_his_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCR_his_hgt_regress.nc")
NCR_his_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCR_his_u_regress.nc")
NCR_his_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCR_his_v_regress.nc")

NCR_ssp585_p3_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/NCR_ssp585_p3_hgt_regress.nc")
NCR_ssp585_p3_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/NCR_ssp585_p3_u_regress.nc")
NCR_ssp585_p3_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/NCR_ssp585_p3_v_regress.nc")

NCRGPCP_ERA5_hgt_slope = NCRGPCP_ERA5_hgt_regress["slope"]
NCRGPCP_ERA5_u_slope = NCRGPCP_ERA5_u_regress["slope"]
NCRGPCP_ERA5_v_slope = NCRGPCP_ERA5_v_regress["slope"]
NCRGPCP_ERA5_hgt_rvalue = NCRGPCP_ERA5_hgt_regress["rvalue"]
NCRGPCP_ERA5_u_rvalue = NCRGPCP_ERA5_u_regress["rvalue"]
NCRGPCP_ERA5_v_rvalue = NCRGPCP_ERA5_v_regress["rvalue"]
NCRGPCP_ERA5_hgt_pvalue = NCRGPCP_ERA5_hgt_regress["pvalue"]
NCRGPCP_ERA5_u_pvalue = NCRGPCP_ERA5_u_regress["pvalue"]
NCRGPCP_ERA5_v_pvalue = NCRGPCP_ERA5_v_regress["pvalue"]

NCR_his_hgt_slope = NCR_his_hgt_regress["slope"]
NCR_his_u_slope = NCR_his_u_regress["slope"]
NCR_his_v_slope = NCR_his_v_regress["slope"]
NCR_his_hgt_rvalue = NCR_his_hgt_regress["rvalue"]
NCR_his_u_rvalue = NCR_his_u_regress["rvalue"]
NCR_his_v_rvalue = NCR_his_v_regress["rvalue"]
NCR_his_hgt_pvalue = NCR_his_hgt_regress["pvalue"]
NCR_his_u_pvalue = NCR_his_u_regress["pvalue"]
NCR_his_v_pvalue = NCR_his_v_regress["pvalue"]

NCR_ssp585_p3_hgt_slope = NCR_ssp585_p3_hgt_regress["slope"]
NCR_ssp585_p3_u_slope = NCR_ssp585_p3_u_regress["slope"]
NCR_ssp585_p3_v_slope = NCR_ssp585_p3_v_regress["slope"]
NCR_ssp585_p3_hgt_rvalue = NCR_ssp585_p3_hgt_regress["rvalue"]
NCR_ssp585_p3_u_rvalue = NCR_ssp585_p3_u_regress["rvalue"]
NCR_ssp585_p3_v_rvalue = NCR_ssp585_p3_v_regress["rvalue"]
NCR_ssp585_p3_hgt_pvalue = NCR_ssp585_p3_hgt_regress["pvalue"]
NCR_ssp585_p3_u_pvalue = NCR_ssp585_p3_u_regress["pvalue"]
NCR_ssp585_p3_v_pvalue = NCR_ssp585_p3_v_regress["pvalue"]
# %%
#   calculate the windcheck and ensmean

NCRGPCP_ERA5_wind_mask = ca.wind_check(
    xr.where(NCRGPCP_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(NCRGPCP_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(NCRGPCP_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(NCRGPCP_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
)

NCR_his_wind_mask = ca.wind_check(
    xr.where(NCR_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(NCR_his_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(NCR_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(NCR_his_v_pvalue <= 0.05, 1.0, 0.0),
)

NCR_ssp585_p3_wind_mask = ca.wind_check(
    xr.where(NCR_ssp585_p3_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(NCR_ssp585_p3_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(NCR_ssp585_p3_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(NCR_ssp585_p3_v_pvalue <= 0.05, 1.0, 0.0),
)

NCR_his_hgt_slope_ens = NCR_his_hgt_slope.mean(dim="models", skipna=True)
NCR_his_hgt_slope_ens_mask = xr.where((ca.MME_reg_mask(NCR_his_hgt_slope_ens, NCR_his_hgt_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(NCR_his_hgt_slope)) >= 2.0, 1.0, 0.0)

NCR_ssp585_p3_hgt_slope_ens = NCR_ssp585_p3_hgt_slope.mean(dim="models", skipna=True)
NCR_ssp585_p3_hgt_slope_ens_mask = xr.where((ca.MME_reg_mask(NCR_ssp585_p3_hgt_slope_ens, NCR_ssp585_p3_hgt_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(NCR_ssp585_p3_hgt_slope)) >= 2.0, 1.0, 0.0)

NCR_his_u_slope_ens = NCR_his_u_slope.mean(dim="models", skipna=True)
NCR_his_u_slope_ens_mask = xr.where((ca.MME_reg_mask(NCR_his_u_slope_ens, NCR_his_u_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(NCR_his_u_slope)) >= 2.0, 1.0, 0.0)

NCR_ssp585_p3_u_slope_ens = NCR_ssp585_p3_u_slope.mean(dim="models", skipna=True)
NCR_ssp585_p3_u_slope_ens_mask = xr.where((ca.MME_reg_mask(NCR_ssp585_p3_u_slope_ens, NCR_ssp585_p3_u_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(NCR_ssp585_p3_u_slope)) >= 2.0, 1.0, 0.0)

NCR_his_v_slope_ens = NCR_his_v_slope.mean(dim="models", skipna=True)
NCR_his_v_slope_ens_mask = xr.where((ca.MME_reg_mask(NCR_his_v_slope_ens, NCR_his_v_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(NCR_his_v_slope)) >= 2.0, 1.0, 0.0)

NCR_ssp585_p3_v_slope_ens = NCR_ssp585_p3_v_slope.mean(dim="models", skipna=True)
NCR_ssp585_p3_v_slope_ens_mask = xr.where((ca.MME_reg_mask(NCR_ssp585_p3_v_slope_ens, NCR_ssp585_p3_v_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(NCR_ssp585_p3_v_slope)) >= 2.0, 1.0, 0.0)

NCR_his_hgt_rvalue_ens = ca.cal_rMME(NCR_his_hgt_rvalue, "models")
NCR_his_hgt_rvalue_ens_mask = xr.where((ca.MME_reg_mask(NCR_his_hgt_rvalue_ens, NCR_his_hgt_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(NCR_his_hgt_slope)) >= 2.0, 1.0, 0.0)

NCR_ssp585_p3_hgt_rvalue_ens = ca.cal_rMME(NCR_ssp585_p3_hgt_rvalue, "models")
NCR_ssp585_p3_hgt_rvalue_ens_mask = xr.where((ca.MME_reg_mask(NCR_ssp585_p3_hgt_rvalue_ens, NCR_ssp585_p3_hgt_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(NCR_ssp585_p3_hgt_slope)) >= 2.0, 1.0, 0.0)

NCR_his_u_rvalue_ens = ca.cal_rMME(NCR_his_u_rvalue, "models")
NCR_his_u_rvalue_ens_mask = xr.where((ca.MME_reg_mask(NCR_his_u_rvalue_ens, NCR_his_u_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(NCR_his_u_slope)) >= 2.0, 1.0, 0.0)

NCR_ssp585_p3_u_rvalue_ens = ca.cal_rMME(NCR_ssp585_p3_u_rvalue, "models")
NCR_ssp585_p3_u_rvalue_ens_mask = xr.where((ca.MME_reg_mask(NCR_ssp585_p3_u_rvalue_ens, NCR_ssp585_p3_u_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(NCR_ssp585_p3_u_slope)) >= 2.0, 1.0, 0.0)

NCR_his_v_rvalue_ens = ca.cal_rMME(NCR_his_v_rvalue, "models")
NCR_his_v_rvalue_ens_mask = xr.where((ca.MME_reg_mask(NCR_his_v_rvalue_ens, NCR_his_v_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(NCR_his_v_slope)) >= 2.0, 1.0, 0.0)

NCR_ssp585_p3_v_rvalue_ens = ca.cal_rMME(NCR_ssp585_p3_v_rvalue, "models")
NCR_ssp585_p3_v_rvalue_ens_mask = xr.where((ca.MME_reg_mask(NCR_ssp585_p3_v_rvalue_ens, NCR_ssp585_p3_v_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(NCR_ssp585_p3_v_slope)) >= 2.0, 1.0, 0.0)

NCR_his_wind_ens_mask = ca.wind_check(
    xr.where(NCR_his_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(NCR_his_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(NCR_his_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(NCR_his_v_slope_ens_mask > 0.0, 1.0, 0.0),
)
NCR_ssp585_p3_wind_ens_mask = ca.wind_check(
    xr.where(NCR_ssp585_p3_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(NCR_ssp585_p3_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(NCR_ssp585_p3_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(NCR_ssp585_p3_v_slope_ens_mask > 0.0, 1.0, 0.0),
)
# %%
#   plot the circulation regress onto NC
#   only plot the circulation regress onto AIR and NCR in MME
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 7), (3, 2))
# plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ======================================
for ax in axs:
    rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
    ax.add_patch(rect)
    # India area
    x0 = India_W
    y0 = India_S
    width = India_E-India_W
    height = India_N-India_S
    patches(ax, x0 - cl, y0, width, height, proj)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    patches(ax, x0 - cl, y0, width, height, proj)
    # SJ-KP area
    x0 = SJ_W
    y0 = SJ_S
    width = SJ_E-SJ_W
    height = SJ_N-SJ_S
    patches(ax, x0 - cl, y0, width, height, proj)
# ======================================
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    # if lev == 200.0:
    #     for ax in axs[num_lev, :]:
    #         x0 = 50
    #         y0 = 15
    #         width = 90
    #         height = 32.5
    #         sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
    # elif lev == 850.0:
    #     for ax in axs[num_lev, :]:
    #         x0 = 110
    #         y0 = 15
    #         width = 27
    #         height = 22.5
    #         sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
    con = axs[num_lev, 0].contourf(
        NCRGPCP_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        NCRGPCP_ERA5_hgt_slope.sel(level=lev), axs[num_lev, 0], n, np.where(NCRGPCP_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_lev, 0].quiver(
        NCRGPCP_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        NCRGPCP_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev, 0].quiver(
        NCRGPCP_ERA5_u_slope.sel(level=lev).where(NCRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        NCRGPCP_ERA5_v_slope.sel(level=lev).where(NCRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 0].format(
        rtitle="1979-2014 {:.0f}hPa".format(lev), ltitle="ERA5",
    )
    # ======================================
    con = axs[num_lev, 1].contourf(
        NCR_his_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        NCR_his_hgt_slope_ens.sel(level=lev), axs[num_lev, 1], n, np.where(NCR_his_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 1].quiver(
        NCR_his_u_slope_ens.sel(level=lev)[::ski, ::ski],
        NCR_his_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev, 1].quiver(
        NCR_his_u_slope_ens.sel(level=lev).where(NCR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        NCR_his_v_slope_ens.sel(level=lev).where(NCR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 1].format(
        rtitle="1979-2014 {:.0f}hPa".format(lev), ltitle="MME",
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="hgt&U reg NCR".format(lev))
# %%
