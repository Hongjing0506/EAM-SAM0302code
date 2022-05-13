'''
Author: ChenHJ
Date: 2022-05-06 15:24:33
LastEditors: ChenHJ
LastEditTime: 2022-05-13 13:51:03
FilePath: /chenhj/0302code/choose_India_area.py
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
fpreCRU = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/cru_ts4.01_r144x72_195001-201412.nc"
)
preCRU = fpreCRU["pre"]
preCRU_JJA = ca.p_time(preCRU, 6, 8, True)/30.67
preCRU_JJA = ca.detrend_dim(preCRU_JJA, "time", deg=1, demean=False)
preCRU_JJA.attrs["units"] = "mm/day"

fpreGPCP = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]
preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)
preGPCP_JJA = ca.detrend_dim(preGPCP_JJA, "time", deg=1, demean=False)

preAIR = xr.open_dataarray("/home/ys17-23/Extension/All_India_Rainfall_index/AIR_mmperday.nc")
preAIR_JJA = ca.p_time(preAIR, 6, 8, True)
preAIR_JJA = preAIR_JJA.sel(time=(preAIR_JJA.time.dt.year>=1950) & (preAIR_JJA.time.dt.year <=2014))
preAIR_JJA = ca.detrend_dim(preAIR_JJA, "time", deg=1, demean=False)

fprehis = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/pr_historical_r144x72_195001-201412.nc")
prehis_JJA = fprehis["pr"]
prehis_JJA.attrs["units"] = "mm/day"
prehis_JJA.attrs["standard_name"] = "precipitation"

fpressp585 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/pr_ssp585_r144x72_201501-209912.nc")
pressp585_JJA = fpressp585["pr"]
pressp585_JJA.attrs["units"] = "mm/day"
pressp585_JJA.attrs["standard_name"] = "precipitation"

fhgtERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/hgt_mon_r144x72_195001-201412.nc")
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

fwERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/omega_mon_r144x72_195001-201412.nc")
wERA5 = fwERA5["w"]

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
hgthis_ver_JJA = fhgthis_ver_JJA["zg"]
hgthis_ver_JJA = hgthis_ver_JJA - hgthis_ver_JJA.mean(dim="lon", skipna=True)

fuhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ua_historical_r144x72_195001-201412.nc")
uhis_ver_JJA = fuhis_ver_JJA["ua"]

fvhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/va_historical_r144x72_195001-201412.nc")
vhis_ver_JJA = fvhis_ver_JJA["va"]

fwhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/wap_historical_r144x72_195001-201412.nc") 
whis_ver_JJA = fwhis_ver_JJA["wap"]

fhgtssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/zg_ssp585_r144x72_201501-209912.nc")
hgtssp585_ver_JJA = fhgtssp585_ver_JJA["zg"]
hgtssp585_ver_JJA = hgtssp585_ver_JJA - hgtssp585_ver_JJA.mean(dim="lon", skipna=True)

fussp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ua_ssp585_r144x72_201501-209912.nc")
ussp585_ver_JJA = fussp585_ver_JJA["ua"]

fvssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/va_ssp585_r144x72_201501-209912.nc")
vssp585_ver_JJA = fvssp585_ver_JJA["va"]

fwssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/wap_ssp585_r144x72_201501-209912.nc")
wssp585_ver_JJA = fwssp585_ver_JJA["wap"]
#read the temperature data in ERA5/historical/ssp585
ftERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/temp_mon_r144x72_195001-201412.nc")
tERA5 = ftERA5["t"]
tERA5_ver_JJA = ca.p_time(tERA5, 6, 8, True)
tERA5_ver_JJA = ca.detrend_dim(tERA5_ver_JJA, "time", deg=1, demean=False)
fthis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ta_historical_r144x72_195001-201412.nc")
this_ver_JJA = fthis_ver_JJA["ta"]
ftssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ta_ssp585_r144x72_201501-209912.nc")
tssp585_ver_JJA = ftssp585_ver_JJA["ta"]

#   read the his_dpg and ssp585_dpg
his_dsdpg = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_dsdpg500-200.nc")
ssp585_dsdpg = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_dsdpg500-200.nc")

#   calculate the utt in historical and ssp585
utthis_JJA = (this_ver_JJA.loc[:,:,500.0:200.0,:,:]*his_dsdpg.data).sum(dim="level",skipna=True)
uttssp585_JJA = (tssp585_ver_JJA.loc[:,:,500.0:200.0,:,:]*ssp585_dsdpg.data).sum(dim="level",skipna=True)
utthis_JJA = ca.detrend_dim(utthis_JJA, "time", deg=1, demean=False)
uttssp585_JJA = ca.detrend_dim(uttssp585_JJA, "time", deg=1, demean=False)
utthis_JJA.name="utt"
uttssp585_JJA.name="utt"

#   deal with the time index for CRU and GPCP data
preCRU_JJA.coords["time"] = prehis_JJA.coords["time"]
preGPCP_JJA.coords["time"] = prehis_JJA.sel(time=prehis_JJA.time.dt.year>=1979).coords["time"]
preAIR_JJA.coords["time"] = prehis_JJA.coords["time"]
# %%
models = uhis_ver_JJA.coords["models"]
models_array = models.data
# %%
#   calculate the ERA5 upper level troposphere temperature between 500hPa to 200hPa
ptop = 1 * 200
g = 9.8
ERA5_dslevel = uERA5_ver_JJA.coords["level"].loc[200.0:500.0] * 100.0
ERA5_dslevel.attrs["units"] = "Pa"
ERA5_dsdp = geocat.comp.dpres_plevel(ERA5_dslevel, spERA5_JJA, ptop)
ERA5_dsdpg = ERA5_dsdp / g
ERA5_dsdpg.attrs["units"] = "kg/m2"
ERA5_dsdpg.name = "dsdpg"

uttERA5_JJA = (tERA5_ver_JJA.loc[:,200.0:500.0,:,:] * ERA5_dsdpg.data).sum(dim="level", skipna=True)
uttERA5_JJA = ca.detrend_dim(uttERA5_JJA, "time", deg=1, demean=False)
uttERA5_JJA.name = "utt"
# %%
#   pick up the area data
#   calculate the vorticity in ERA5, historical and ssp585
vorERA5_ver_JJA = mpcalc.vorticity(uERA5_ver_JJA.sel(level=200.0), vERA5_ver_JJA.sel(level=200.0))
vorERA5_ver_JJA = vorERA5_ver_JJA.metpy.dequantify()

vorhis_ver_JJA = mpcalc.vorticity(uhis_ver_JJA.sel(level=200.0), vhis_ver_JJA.sel(level=200.0))
vorhis_ver_JJA = vorhis_ver_JJA.metpy.dequantify()

vorssp585_ver_JJA = mpcalc.vorticity(ussp585_ver_JJA.sel(level=200.0), vssp585_ver_JJA.sel(level=200.0))
vorssp585_ver_JJA = vorssp585_ver_JJA.metpy.dequantify()

#   calculate the precipitation in India
lat = preCRU_JJA.coords["lat"]
lon = preCRU_JJA.coords["lon"]

India_N = 32.5
# India_N = 30.0
India_S = 8.0
India_W = 70.0
India_E = 86.0
lat_India_range = lat[(lat >= India_S) & (lat <= India_N)]
lon_India_range = lon[(lon >= India_W) & (lon <= India_E)]

preCRU_India_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
preGPCP_India_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
prehis_India_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
pressp585_India_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Northern China
NC_N = 40.0
NC_S = 32.5
NC_W = 100.0
NC_E = 120.0
lat_NC_range = lat[(lat >= NC_S) & (lat <= NC_N)]
lon_NC_range = lon[(lon >= NC_W) & (lon <= NC_E)]
# lat_NC_range = lat[(lat>=27.5) & (lat<=37.5)]
# lon_NC_range = lon[(lon>=105.0) & (lon<=125.0)]
preCRU_NC_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)
preGPCP_NC_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)
prehis_NC_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)
pressp585_NC_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Southern China
lat_SC_range = lat[(lat>=20.0) & (lat<=27.5)]
lon_SC_range = lon[(lon>=105.0) & (lon<=125.0)]
preCRU_SC_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
preGPCP_SC_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
prehis_SC_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
pressp585_SC_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Korean Peninsula
KP_N = 37.5
KP_S = 32.5
KP_W = 124.0
KP_E = 132.0
lat_KP_range = lat[(lat >= KP_S) & (lat <= KP_N)]
lon_KP_range = lon[(lon >= KP_W) & (lon <= KP_E)]
# lat_KP_range = lat[(lat>=27.5) & (lat<=37.5)]
# lon_KP_range = lon[(lon>=105.0) & (lon<=125.0)]
preCRU_KP_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)
preGPCP_KP_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)
prehis_KP_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)
pressp585_KP_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_KP_range, lon=lon_KP_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Korean Peninsula-Southern Japan
SJ_N = 36.0
SJ_S = 31.0
SJ_W = 124.0
SJ_E = 136.0
lat_SJ_range = lat[(lat >= SJ_S) & (lat <= SJ_N)]
lon_SJ_range = lon[(lon >= SJ_W) & (lon <= SJ_E)]
# lat_SJ_range = lat[(lat>=27.5) & (lat<=37.5)]
# lon_SJ_range = lon[(lon>=105.0) & (lon<=125.0)]
preCRU_SJ_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)
preGPCP_SJ_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)
prehis_SJ_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)
pressp585_SJ_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)

#   calculate the 200hPa u-wind over the East Asia
lat_EA_range = lat[(lat>=30.0) & (lat<=40.0)]
lon_EA_range = lon[(lon>=100.0) & (lon<=120.0)]

uERA5_EA_JJA = ca.cal_lat_weighted_mean(uERA5_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)
uhis_EA_JJA = ca.cal_lat_weighted_mean(uhis_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)
ussp585_EA_JJA = ca.cal_lat_weighted_mean(ussp585_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)

#   calculate the 200hPa vorticity over the East Asia
lat_EAhigh_range = lat[(lat>=25.0) & (lat<=50.0)]
lon_EAhigh_range = lon[(lon>=105.0) & (lon<=135.0)]
uERA5_EAhigh_JJA = uERA5_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
uhis_EAhigh_JJA = uhis_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
ussp585_EAhigh_JJA = ussp585_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)

vERA5_EAhigh_JJA = vERA5_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
vhis_EAhigh_JJA = vhis_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
vssp585_EAhigh_JJA = vssp585_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)

vorERA5_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uERA5_EAhigh_JJA, vERA5_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorhis_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uhis_EAhigh_JJA, vhis_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_EAhigh_JJA, vssp585_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()

#   calculate the longitude mean over 100.0° to 125°E
lon_EA_range = lon[(lon>=100.0)&(lon<=125.0)]

uERA5_EA_lm_JJA = uERA5_ver_JJA.loc[:,10.0:,0.:,:].sel(lon=lon_EA_range).mean(dim="lon",skipna=True)
uhis_EA_lm_JJA = uhis_ver_JJA.loc[:,:,:10.0,0.:,:].sel(lon=lon_EA_range).mean(dim="lon",skipna=True)
ussp585_EA_lm_JJA = ussp585_ver_JJA.loc[:,:,:10.0,0.:,:].sel(lon=lon_EA_range).mean(dim="lon",skipna=True)

tERA5_EA_lm_JJA = tERA5_ver_JJA.loc[:,10.0:,0.:,:].sel(lon=lon_EA_range).mean(dim="lon",skipna=True)
this_EA_lm_JJA = this_ver_JJA.loc[:,:,:10.0,0.:,:].sel(lon=lon_EA_range).mean(dim="lon",skipna=True)
tssp585_EA_lm_JJA = tssp585_ver_JJA.loc[:,:,:10.0,0.:,:].sel(lon=lon_EA_range).mean(dim="lon",skipna=True)

#   calculate the MTG related to the 200hPa u wind over East Asia
lat_area1_range = lat[(lat >= 15.0) & (lat <= 30.0)]
lon_area1_range = lon[(lon >= 100.0) & (lon <= 125.0)]

lat_area2_range = lat[(lat >= 33.75) & (lat <= 45.0)]
lon_area2_range = lon[(lon >= 100.0) & (lon <= 125.0)]

EAU_MTGERA5_JJA = ca.cal_lat_weighted_mean(uttERA5_JJA.sel(lat=lat_area2_range,lon=lon_area2_range)).mean(dim="lon",skipna=True)-ca.cal_lat_weighted_mean(uttERA5_JJA.sel(lat=lat_area1_range,lon=lon_area1_range)).mean(dim="lon",skipna=True)
EAU_MTGhis_JJA = ca.cal_lat_weighted_mean(utthis_JJA.sel(lat=lat_area2_range,lon=lon_area2_range)).mean(dim="lon",skipna=True)-ca.cal_lat_weighted_mean(utthis_JJA.sel(lat=lat_area1_range,lon=lon_area1_range)).mean(dim="lon",skipna=True)
EAU_MTGssp585_JJA = ca.cal_lat_weighted_mean(uttssp585_JJA.sel(lat=lat_area2_range,lon=lon_area2_range)).mean(dim="lon",skipna=True)-ca.cal_lat_weighted_mean(uttssp585_JJA.sel(lat=lat_area1_range,lon=lon_area1_range)).mean(dim="lon",skipna=True)

#   calculate the vorticity over the West Asia in 200hPa
lat_WAhigh_range = lat[(lat>=25.0) & (lat<=45.0)]
lon_WAhigh_range = lon[(lon>=55.0) & (lon<=75.0)]
uERA5_WAhigh_JJA = uERA5_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
uhis_WAhigh_JJA = uhis_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
ussp585_WAhigh_JJA = ussp585_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)

vERA5_WAhigh_JJA = vERA5_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
vhis_WAhigh_JJA = vhis_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
vssp585_WAhigh_JJA = vssp585_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)

vorERA5_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uERA5_WAhigh_JJA, vERA5_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorhis_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uhis_WAhigh_JJA, vhis_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_WAhigh_JJA, vssp585_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
# %%
#   calculate the detrend-data for the picked-up area data
vorERA5_ver_JJA = ca.detrend_dim(vorERA5_ver_JJA, "time", deg=1, demean=False)
vorhis_ver_JJA = ca.detrend_dim(vorhis_ver_JJA, "time", deg=1, demean=False)
vorssp585_ver_JJA = ca.detrend_dim(vorssp585_ver_JJA, "time", deg=1, demean=False)

preCRU_India_JJA = ca.detrend_dim(preCRU_India_JJA, "time", deg=1, demean=False)
preGPCP_India_JJA = ca.detrend_dim(preGPCP_India_JJA, "time", deg=1, demean=False)
prehis_India_JJA = ca.detrend_dim(prehis_India_JJA, "time", deg=1, demean=False)
pressp585_India_JJA = ca.detrend_dim(pressp585_India_JJA, "time", deg=1, demean=False)

preCRU_NC_JJA = ca.detrend_dim(preCRU_NC_JJA, "time", deg=1, demean=False)
preGPCP_NC_JJA = ca.detrend_dim(preGPCP_NC_JJA, "time", deg=1, demean=False)
prehis_NC_JJA = ca.detrend_dim(prehis_NC_JJA, "time", deg=1, demean=False)
pressp585_NC_JJA = ca.detrend_dim(pressp585_NC_JJA, "time", deg=1, demean=False)

preCRU_KP_JJA = ca.detrend_dim(preCRU_KP_JJA, "time", deg=1, demean=False)
preGPCP_KP_JJA = ca.detrend_dim(preGPCP_KP_JJA, "time", deg=1, demean=False)
prehis_KP_JJA = ca.detrend_dim(prehis_KP_JJA, "time", deg=1, demean=False)
pressp585_KP_JJA = ca.detrend_dim(pressp585_KP_JJA, "time", deg=1, demean=False)

preCRU_SJ_JJA = ca.detrend_dim(preCRU_SJ_JJA, "time", deg=1, demean=False)
preGPCP_SJ_JJA = ca.detrend_dim(preGPCP_SJ_JJA, "time", deg=1, demean=False)
prehis_SJ_JJA = ca.detrend_dim(prehis_SJ_JJA, "time", deg=1, demean=False)
pressp585_SJ_JJA = ca.detrend_dim(pressp585_SJ_JJA, "time", deg=1, demean=False)

preCRU_SC_JJA = ca.detrend_dim(preCRU_SC_JJA, "time", deg=1, demean=False)
preGPCP_SC_JJA = ca.detrend_dim(preGPCP_SC_JJA, "time", deg=1, demean=False)
prehis_SC_JJA = ca.detrend_dim(prehis_SC_JJA, "time", deg=1, demean=False)
pressp585_SC_JJA = ca.detrend_dim(pressp585_SC_JJA, "time", deg=1, demean=False)

uERA5_EA_JJA = ca.detrend_dim(uERA5_EA_JJA, "time", deg=1, demean=False)
uhis_EA_JJA = ca.detrend_dim(uhis_EA_JJA, "time", deg=1, demean=False)
ussp585_EA_JJA = ca.detrend_dim(ussp585_EA_JJA, "time", deg=1, demean=False)

vorERA5_EAhigh_JJA = ca.detrend_dim(vorERA5_EAhigh_JJA, "time", deg=1, demean=False)
vorhis_EAhigh_JJA = ca.detrend_dim(vorhis_EAhigh_JJA, "time", deg=1, demean=False)
vorssp585_EAhigh_JJA = ca.detrend_dim(vorssp585_EAhigh_JJA, "time", deg=1, demean=False)

uERA5_EA_lm_JJA = ca.detrend_dim(uERA5_EA_lm_JJA, "time", deg=1, demean=False)
uhis_EA_lm_JJA = ca.detrend_dim(uhis_EA_lm_JJA, "time", deg=1, demean=False)
ussp585_EA_lm_JJA = ca.detrend_dim(ussp585_EA_lm_JJA, "time", deg=1, demean=False)

tERA5_EA_lm_JJA = ca.detrend_dim(tERA5_EA_lm_JJA, "time", deg=1, demean=False)
this_EA_lm_JJA = ca.detrend_dim(this_EA_lm_JJA, "time", deg=1, demean=False)
tssp585_EA_lm_JJA = ca.detrend_dim(tssp585_EA_lm_JJA, "time", deg=1, demean=False)

EAU_MTGhis_JJA = ca.detrend_dim(EAU_MTGhis_JJA, "time", deg=1, demean=False)
EAU_MTGssp585_JJA = ca.detrend_dim(EAU_MTGssp585_JJA, "time", deg=1, demean=False)
EAU_MTGERA5_JJA = ca.detrend_dim(EAU_MTGERA5_JJA, "time", deg=1, demean=False)

vorERA5_WAhigh_JJA = ca.detrend_dim(vorERA5_WAhigh_JJA, "time", deg=1, demean=False)
vorhis_WAhigh_JJA = ca.detrend_dim(vorhis_WAhigh_JJA, "time", deg=1, demean=False)
vorssp585_WAhigh_JJA = ca.detrend_dim(vorssp585_WAhigh_JJA, "time", deg=1, demean=False)
# %%
#   calculate the precipitation fields regression onto IndR
(
    pre_CRU_India_pre_slope,
    pre_CRU_India_pre_intercept,
    pre_CRU_India_pre_rvalue,
    pre_CRU_India_pre_pvalue,
    pre_CRU_India_pre_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA.sel(time=preCRU_India_JJA.time.dt.year>=1979), preCRU_JJA.sel(time=preCRU_JJA.time.dt.year>=1979))

(
    pre_GPCP_India_pre_slope,
    pre_GPCP_India_pre_intercept,
    pre_GPCP_India_pre_rvalue,
    pre_GPCP_India_pre_pvalue,
    pre_GPCP_India_pre_hypothesis,
) = ca.dim_linregress(preGPCP_India_JJA, preGPCP_JJA)

preAIR_JJA.coords["time"] = preCRU_JJA.coords["time"]
(
    pre_AIR_India_pre_slope,
    pre_AIR_India_pre_intercept,
    pre_AIR_India_pre_rvalue,
    pre_AIR_India_pre_pvalue,
    pre_AIR_India_pre_hypothesis,
) = ca.dim_linregress(preAIR_JJA.sel(time=preAIR_JJA.time.dt.year>=1979), preGPCP_JJA.sel(time=preGPCP_JJA.time.dt.year>=1979))

(
    pre_his_India_pre_slope,
    pre_his_India_pre_intercept,
    pre_his_India_pre_rvalue,
    pre_his_India_pre_pvalue,
    pre_his_India_pre_hypothesis,
) = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), prehis_JJA.sel(time=prehis_JJA.time.dt.year>=1979))

(
    pre_ssp585_p3_India_pre_slope,
    pre_ssp585_p3_India_pre_intercept,
    pre_ssp585_p3_India_pre_rvalue,
    pre_ssp585_p3_India_pre_pvalue,
    pre_ssp585_p3_India_pre_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), pressp585_JJA.sel(time=pressp585_JJA.time.dt.year>=2064))
# %%
#   calculate the MME for historical and ssp585_p3
pre_his_India_pre_slope_ens = pre_his_India_pre_slope.mean(dim="models", skipna=True)

pre_his_India_pre_slope_ens_mask = xr.where((ca.MME_reg_mask(pre_his_India_pre_slope_ens, pre_his_India_pre_slope.std(dim="models", skipna=True), len(pre_his_India_pre_slope.coords["models"]), True) + ca.cal_mmemask(pre_his_India_pre_slope)) >= 2.0, 1.0, 0.0)

pre_his_India_pre_rvalue_ens = ca.cal_rMME(pre_his_India_pre_rvalue,"models")

pre_his_India_pre_rvalue_ens_mask = xr.where((ca.MME_reg_mask(pre_his_India_pre_rvalue_ens, pre_his_India_pre_rvalue.std(dim="models", skipna=True), len(pre_his_India_pre_rvalue.coords["models"]), True) + ca.cal_mmemask(pre_his_India_pre_slope)) >= 2.0, 1.0, 0.0)

pre_ssp585_p3_India_pre_slope_ens = pre_ssp585_p3_India_pre_slope.mean(dim="models", skipna=True)

pre_ssp585_p3_India_pre_slope_ens_mask = xr.where((ca.MME_reg_mask(pre_ssp585_p3_India_pre_slope_ens, pre_ssp585_p3_India_pre_slope.std(dim="models", skipna=True), len(pre_ssp585_p3_India_pre_slope.coords["models"]), True) + ca.cal_mmemask(pre_ssp585_p3_India_pre_slope)) >= 2.0, 1.0, 0.0)

pre_ssp585_p3_India_pre_rvalue_ens = ca.cal_rMME(pre_ssp585_p3_India_pre_rvalue,"models")

pre_ssp585_p3_India_pre_rvalue_ens_mask = xr.where((ca.MME_reg_mask(pre_ssp585_p3_India_pre_rvalue_ens, pre_ssp585_p3_India_pre_rvalue.std(dim="models", skipna=True), len(pre_ssp585_p3_India_pre_rvalue.coords["models"]), True) + ca.cal_mmemask(pre_ssp585_p3_India_pre_slope)) >= 2.0, 1.0, 0.0)
# %%
#   plot the correlation coefficients rvalue in CRU, GPCP, historical run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (6, 5))
plot_array[5,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
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
# ===================================================
con = axs[0].contourf(
    pre_CRU_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_CRU_India_pre_rvalue, axs[0], n, np.where(pre_CRU_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="CRU",
)
# ===================================================
con = axs[1].contourf(
    pre_GPCP_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_GPCP_India_pre_rvalue, axs[1], n, np.where(pre_GPCP_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="GPCP",
)
# # ===================================================
con = axs[2].contourf(
    pre_his_India_pre_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_his_India_pre_rvalue_ens, axs[2], n, np.where(pre_his_India_pre_rvalue_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[2].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_his_India_pre_slope.coords["models"].data):
    con = axs[num_models+3].contourf(
    pre_his_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_his_India_pre_rvalue.sel(models=mod), axs[num_models+3], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.10), "bright purple", 4.0,
    )

    axs[num_models+3].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   calculate the hgt/u/v regression onto IndR in ERA5, historical, ssp585, ssp585_p3
preCRU_India_JJA.coords["time"] = hgtERA5_ver_JJA.coords["time"]
preGPCP_India_JJA.coords["time"] = hgtERA5_ver_JJA.sel(time=hgtERA5_ver_JJA.time.dt.year>=1979).coords["time"]
preAIR_JJA.coords["time"] = hgtERA5_ver_JJA.coords["time"]
(
    IndRCRU_ERA5_hgt_slope,
    IndRCRU_ERA5_hgt_intercept,
    IndRCRU_ERA5_hgt_rvalue,
    IndRCRU_ERA5_hgt_pvalue,
    IndRCRU_ERA5_hgt_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA.sel(time=preCRU_India_JJA.time.dt.year>=1979), hgtERA5_ver_JJA.sel(time=hgtERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRCRU_ERA5_u_slope,
    IndRCRU_ERA5_u_intercept,
    IndRCRU_ERA5_u_rvalue,
    IndRCRU_ERA5_u_pvalue,
    IndRCRU_ERA5_u_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA.sel(time=preCRU_India_JJA.time.dt.year>=1979), uERA5_ver_JJA.sel(time=uERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRCRU_ERA5_v_slope,
    IndRCRU_ERA5_v_intercept,
    IndRCRU_ERA5_v_rvalue,
    IndRCRU_ERA5_v_pvalue,
    IndRCRU_ERA5_v_hypothesis,
) = ca.dim_linregress(preCRU_India_JJA.sel(time=preCRU_India_JJA.time.dt.year>=1979), vERA5_ver_JJA.sel(time=vERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRGPCP_ERA5_hgt_slope,
    IndRGPCP_ERA5_hgt_intercept,
    IndRGPCP_ERA5_hgt_rvalue,
    IndRGPCP_ERA5_hgt_pvalue,
    IndRGPCP_ERA5_hgt_hypothesis,
) = ca.dim_linregress(preGPCP_India_JJA.sel(time=preGPCP_India_JJA.time.dt.year>=1979), hgtERA5_ver_JJA.sel(time=hgtERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRGPCP_ERA5_u_slope,
    IndRGPCP_ERA5_u_intercept,
    IndRGPCP_ERA5_u_rvalue,
    IndRGPCP_ERA5_u_pvalue,
    IndRGPCP_ERA5_u_hypothesis,
) = ca.dim_linregress(preGPCP_India_JJA.sel(time=preGPCP_India_JJA.time.dt.year>=1979), uERA5_ver_JJA.sel(time=uERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRGPCP_ERA5_v_slope,
    IndRGPCP_ERA5_v_intercept,
    IndRGPCP_ERA5_v_rvalue,
    IndRGPCP_ERA5_v_pvalue,
    IndRGPCP_ERA5_v_hypothesis,
) = ca.dim_linregress(preGPCP_India_JJA.sel(time=preGPCP_India_JJA.time.dt.year>=1979), vERA5_ver_JJA.sel(time=vERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndR_his_hgt_slope,
    IndR_his_hgt_intercept,
    IndR_his_hgt_rvalue,
    IndR_his_hgt_pvalue,
    IndR_his_hgt_hypothesis,
) = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), hgthis_ver_JJA.sel(time=hgthis_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndR_his_u_slope,
    IndR_his_u_intercept,
    IndR_his_u_rvalue,
    IndR_his_u_pvalue,
    IndR_his_u_hypothesis,
) = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), uhis_ver_JJA.sel(time=uhis_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndR_his_v_slope,
    IndR_his_v_intercept,
    IndR_his_v_rvalue,
    IndR_his_v_pvalue,
    IndR_his_v_hypothesis,
) = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), vhis_ver_JJA.sel(time=vhis_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndR_ssp585_p3_hgt_slope,
    IndR_ssp585_p3_hgt_intercept,
    IndR_ssp585_p3_hgt_rvalue,
    IndR_ssp585_p3_hgt_pvalue,
    IndR_ssp585_p3_hgt_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), hgtssp585_ver_JJA.sel(time=hgtssp585_ver_JJA.time.dt.year>=2064, level=[200.0, 500.0, 850.0]))

(
    IndR_ssp585_p3_u_slope,
    IndR_ssp585_p3_u_intercept,
    IndR_ssp585_p3_u_rvalue,
    IndR_ssp585_p3_u_pvalue,
    IndR_ssp585_p3_u_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), ussp585_ver_JJA.sel(time=ussp585_ver_JJA.time.dt.year>=2064, level=[200.0, 500.0, 850.0]))

(
    IndR_ssp585_p3_v_slope,
    IndR_ssp585_p3_v_intercept,
    IndR_ssp585_p3_v_rvalue,
    IndR_ssp585_p3_v_pvalue,
    IndR_ssp585_p3_v_hypothesis,
) = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), vssp585_ver_JJA.sel(time=vssp585_ver_JJA.time.dt.year>=2064, level=[200.0, 500.0, 850.0]))

(
    IndRAIR_ERA5_hgt_slope,
    IndRAIR_ERA5_hgt_intercept,
    IndRAIR_ERA5_hgt_rvalue,
    IndRAIR_ERA5_hgt_pvalue,
    IndRAIR_ERA5_hgt_hypothesis,
) = ca.dim_linregress(preAIR_JJA.sel(time=preAIR_JJA.time.dt.year>=1979), hgtERA5_ver_JJA.sel(time=hgtERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRAIR_ERA5_u_slope,
    IndRAIR_ERA5_u_intercept,
    IndRAIR_ERA5_u_rvalue,
    IndRAIR_ERA5_u_pvalue,
    IndRAIR_ERA5_u_hypothesis,
) = ca.dim_linregress(preAIR_JJA.sel(time=preAIR_JJA.time.dt.year>=1979), uERA5_ver_JJA.sel(time=uERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRAIR_ERA5_v_slope,
    IndRAIR_ERA5_v_intercept,
    IndRAIR_ERA5_v_rvalue,
    IndRAIR_ERA5_v_pvalue,
    IndRAIR_ERA5_v_hypothesis,
) = ca.dim_linregress(preAIR_JJA.sel(time=preAIR_JJA.time.dt.year>=1979), vERA5_ver_JJA.sel(time=vERA5_ver_JJA.time.dt.year>=1979, level=[200.0, 500.0, 850.0]))

(
    IndRAIR_ERA5_vor_slope,
    IndRAIR_ERA5_vor_intercept,
    IndRAIR_ERA5_vor_rvalue,
    IndRAIR_ERA5_vor_pvalue,
    IndRAIR_ERA5_vor_hypothesis,
) = ca.dim_linregress(preAIR_JJA.sel(time=preAIR_JJA.time.dt.year>=1979), vorERA5_ver_JJA.sel(time=vorERA5_ver_JJA.time.dt.year>=1979))

(
    IndR_his_vor_slope,
    IndR_his_vor_intercept,
    IndR_his_vor_rvalue,
    IndR_his_vor_pvalue,
    IndR_his_vor_hypothesis,
) = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), vorhis_ver_JJA.sel(time=vorhis_ver_JJA.time.dt.year>=1979))
# %%
#   save the regression results
level=IndR_his_hgt_slope.coords["level"]
lat=IndR_his_hgt_slope.coords["lat"]
lon=IndR_his_hgt_slope.coords["lon"]

IndRCRU_ERA5_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRCRU_ERA5_hgt_slope.data),
        intercept=(["level", "lat", "lon"], IndRCRU_ERA5_hgt_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRCRU_ERA5_hgt_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRCRU_ERA5_hgt_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRCRU_ERA5_hgt_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of ERA5 regress onto 1979-2014 CRU IndR"),
)

IndRCRU_ERA5_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRCRU_ERA5_u_slope.data),
        intercept=(["level", "lat", "lon"], IndRCRU_ERA5_u_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRCRU_ERA5_u_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRCRU_ERA5_u_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRCRU_ERA5_u_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of ERA5 regress onto 1979-2014 CRU IndR"),
)

IndRCRU_ERA5_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRCRU_ERA5_v_slope.data),
        intercept=(["level", "lat", "lon"], IndRCRU_ERA5_v_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRCRU_ERA5_v_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRCRU_ERA5_v_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRCRU_ERA5_v_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of ERA5 regress onto 1979-2014 CRU IndR"),
)

IndRAIR_ERA5_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_slope.data),
        intercept=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of ERA5 regress onto 1979-2014 AIR"),
)

IndRAIR_ERA5_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRAIR_ERA5_u_slope.data),
        intercept=(["level", "lat", "lon"], IndRAIR_ERA5_u_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRAIR_ERA5_u_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRAIR_ERA5_u_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRAIR_ERA5_u_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of ERA5 regress onto 1979-2014 AIR"),
)

IndRAIR_ERA5_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRAIR_ERA5_v_slope.data),
        intercept=(["level", "lat", "lon"], IndRAIR_ERA5_v_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRAIR_ERA5_v_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRAIR_ERA5_v_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRAIR_ERA5_v_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of ERA5 regress onto 1979-2014 AIR"),
)

IndRGPCP_ERA5_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRGPCP_ERA5_hgt_slope.data),
        intercept=(["level", "lat", "lon"], IndRGPCP_ERA5_hgt_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRGPCP_ERA5_hgt_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRGPCP_ERA5_hgt_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRGPCP_ERA5_hgt_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of ERA5 regress onto 1979-2014 GPCP IndR"),
)

IndRGPCP_ERA5_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRGPCP_ERA5_u_slope.data),
        intercept=(["level", "lat", "lon"], IndRGPCP_ERA5_u_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRGPCP_ERA5_u_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRGPCP_ERA5_u_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRGPCP_ERA5_u_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of ERA5 regress onto 1979-2014 GPCP IndR"),
)

IndRGPCP_ERA5_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["level", "lat", "lon"], IndRGPCP_ERA5_v_slope.data),
        intercept=(["level", "lat", "lon"], IndRGPCP_ERA5_v_intercept.data),
        rvalue=(["level", "lat", "lon"], IndRGPCP_ERA5_v_rvalue.data),
        pvalue=(["level", "lat", "lon"], IndRGPCP_ERA5_v_pvalue.data),
        hypothesis=(["level", "lat", "lon"], IndRGPCP_ERA5_v_hypothesis.data),
    ),
    coords=dict(
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of ERA5 regress onto 1979-2014 GPCP IndR"),
)

IndR_his_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_his_hgt_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_his_hgt_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_his_hgt_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_his_hgt_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_his_hgt_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of historical regress onto 1979-2014 IndR"),
)

IndR_his_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_his_u_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_his_u_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_his_u_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_his_u_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_his_u_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of historical regress onto 1979-2014 IndR"),
)

IndR_his_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_his_v_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_his_v_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_his_v_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_his_v_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_his_v_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of historical regress onto 1979-2014 IndR"),
)

IndR_ssp585_p3_hgt_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="hgt fields of ssp585_p3 regress onto 2064-2099 IndR"),
)

IndR_ssp585_p3_u_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="u fields of ssp585_p3 regress onto 2064-2099 IndR"),
)

IndR_ssp585_p3_v_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_slope.data),
        intercept=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_intercept.data),
        rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_rvalue.data),
        pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_pvalue.data),
        hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_hypothesis.data),
    ),
    coords=dict(
        models=models.data,
        level=level.data,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="v fields of ssp585_p3 regress onto 2064-2099 IndR"),
)

IndRCRU_ERA5_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRCRU_ERA5_hgt_regress.nc")
IndRCRU_ERA5_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRCRU_ERA5_u_regress.nc")
IndRCRU_ERA5_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRCRU_ERA5_v_regress.nc")

IndRAIR_ERA5_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRAIR_ERA5_hgt_regress.nc")
IndRAIR_ERA5_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRAIR_ERA5_u_regress.nc")
IndRAIR_ERA5_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRAIR_ERA5_v_regress.nc")

IndRGPCP_ERA5_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRGPCP_ERA5_hgt_regress.nc")
IndRGPCP_ERA5_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRGPCP_ERA5_u_regress.nc")
IndRGPCP_ERA5_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRGPCP_ERA5_v_regress.nc")

IndR_his_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_hgt_regress.nc")
IndR_his_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_u_regress.nc")
IndR_his_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_v_regress.nc")

IndR_ssp585_p3_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_hgt_regress.nc")
IndR_ssp585_p3_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_u_regress.nc")
IndR_ssp585_p3_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_v_regress.nc")
# %%
#   read the regression data
IndRCRU_ERA5_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRCRU_ERA5_hgt_regress.nc")
IndRCRU_ERA5_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRCRU_ERA5_u_regress.nc")
IndRCRU_ERA5_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRCRU_ERA5_v_regress.nc")

IndRAIR_ERA5_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRAIR_ERA5_hgt_regress.nc")
IndRAIR_ERA5_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRAIR_ERA5_u_regress.nc")
IndRAIR_ERA5_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRAIR_ERA5_v_regress.nc")

IndRGPCP_ERA5_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRGPCP_ERA5_hgt_regress.nc")
IndRGPCP_ERA5_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRGPCP_ERA5_u_regress.nc")
IndRGPCP_ERA5_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRGPCP_ERA5_v_regress.nc")

IndR_his_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_hgt_regress.nc")
IndR_his_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_u_regress.nc")
IndR_his_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_v_regress.nc")

IndR_ssp585_p3_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_hgt_regress.nc")
IndR_ssp585_p3_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_u_regress.nc")
IndR_ssp585_p3_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_v_regress.nc")

IndRCRU_ERA5_hgt_slope = IndRCRU_ERA5_hgt_regress["slope"]
IndRCRU_ERA5_u_slope = IndRCRU_ERA5_u_regress["slope"]
IndRCRU_ERA5_v_slope = IndRCRU_ERA5_v_regress["slope"]
IndRCRU_ERA5_hgt_rvalue = IndRCRU_ERA5_hgt_regress["rvalue"]
IndRCRU_ERA5_u_rvalue = IndRCRU_ERA5_u_regress["rvalue"]
IndRCRU_ERA5_v_rvalue = IndRCRU_ERA5_v_regress["rvalue"]
IndRCRU_ERA5_hgt_pvalue = IndRCRU_ERA5_hgt_regress["pvalue"]
IndRCRU_ERA5_u_pvalue = IndRCRU_ERA5_u_regress["pvalue"]
IndRCRU_ERA5_v_pvalue = IndRCRU_ERA5_v_regress["pvalue"]

IndRAIR_ERA5_hgt_slope = IndRAIR_ERA5_hgt_regress["slope"]
IndRAIR_ERA5_u_slope = IndRAIR_ERA5_u_regress["slope"]
IndRAIR_ERA5_v_slope = IndRAIR_ERA5_v_regress["slope"]
IndRAIR_ERA5_hgt_rvalue = IndRAIR_ERA5_hgt_regress["rvalue"]
IndRAIR_ERA5_u_rvalue = IndRAIR_ERA5_u_regress["rvalue"]
IndRAIR_ERA5_v_rvalue = IndRAIR_ERA5_v_regress["rvalue"]
IndRAIR_ERA5_hgt_pvalue = IndRAIR_ERA5_hgt_regress["pvalue"]
IndRAIR_ERA5_u_pvalue = IndRAIR_ERA5_u_regress["pvalue"]
IndRAIR_ERA5_v_pvalue = IndRAIR_ERA5_v_regress["pvalue"]

IndRGPCP_ERA5_hgt_slope = IndRGPCP_ERA5_hgt_regress["slope"]
IndRGPCP_ERA5_u_slope = IndRGPCP_ERA5_u_regress["slope"]
IndRGPCP_ERA5_v_slope = IndRGPCP_ERA5_v_regress["slope"]
IndRGPCP_ERA5_hgt_rvalue = IndRGPCP_ERA5_hgt_regress["rvalue"]
IndRGPCP_ERA5_u_rvalue = IndRGPCP_ERA5_u_regress["rvalue"]
IndRGPCP_ERA5_v_rvalue = IndRGPCP_ERA5_v_regress["rvalue"]
IndRGPCP_ERA5_hgt_pvalue = IndRGPCP_ERA5_hgt_regress["pvalue"]
IndRGPCP_ERA5_u_pvalue = IndRGPCP_ERA5_u_regress["pvalue"]
IndRGPCP_ERA5_v_pvalue = IndRGPCP_ERA5_v_regress["pvalue"]

IndR_his_hgt_slope = IndR_his_hgt_regress["slope"]
IndR_his_u_slope = IndR_his_u_regress["slope"]
IndR_his_v_slope = IndR_his_v_regress["slope"]
IndR_his_hgt_rvalue = IndR_his_hgt_regress["rvalue"]
IndR_his_u_rvalue = IndR_his_u_regress["rvalue"]
IndR_his_v_rvalue = IndR_his_v_regress["rvalue"]
IndR_his_hgt_pvalue = IndR_his_hgt_regress["pvalue"]
IndR_his_u_pvalue = IndR_his_u_regress["pvalue"]
IndR_his_v_pvalue = IndR_his_v_regress["pvalue"]

IndR_ssp585_p3_hgt_slope = IndR_ssp585_p3_hgt_regress["slope"]
IndR_ssp585_p3_u_slope = IndR_ssp585_p3_u_regress["slope"]
IndR_ssp585_p3_v_slope = IndR_ssp585_p3_v_regress["slope"]
IndR_ssp585_p3_hgt_rvalue = IndR_ssp585_p3_hgt_regress["rvalue"]
IndR_ssp585_p3_u_rvalue = IndR_ssp585_p3_u_regress["rvalue"]
IndR_ssp585_p3_v_rvalue = IndR_ssp585_p3_v_regress["rvalue"]
IndR_ssp585_p3_hgt_pvalue = IndR_ssp585_p3_hgt_regress["pvalue"]
IndR_ssp585_p3_u_pvalue = IndR_ssp585_p3_u_regress["pvalue"]
IndR_ssp585_p3_v_pvalue = IndR_ssp585_p3_v_regress["pvalue"]
# %%
#   calculate the windcheck and ensmean
IndRCRU_ERA5_wind_mask = ca.wind_check(
    xr.where(IndRCRU_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRCRU_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRCRU_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRCRU_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
)

IndRAIR_ERA5_wind_mask = ca.wind_check(
    xr.where(IndRAIR_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRAIR_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRAIR_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRAIR_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
)

IndRGPCP_ERA5_wind_mask = ca.wind_check(
    xr.where(IndRGPCP_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRGPCP_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRGPCP_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRGPCP_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
)

IndR_his_wind_mask = ca.wind_check(
    xr.where(IndR_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_his_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_his_v_pvalue <= 0.05, 1.0, 0.0),
)

IndR_ssp585_p3_wind_mask = ca.wind_check(
    xr.where(IndR_ssp585_p3_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_pvalue <= 0.05, 1.0, 0.0),
)

IndR_his_hgt_slope_ens = IndR_his_hgt_slope.mean(dim="models", skipna=True)
IndR_his_hgt_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_hgt_slope_ens, IndR_his_hgt_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_hgt_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_hgt_slope_ens = IndR_ssp585_p3_hgt_slope.mean(dim="models", skipna=True)
IndR_ssp585_p3_hgt_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_hgt_slope_ens, IndR_ssp585_p3_hgt_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_hgt_slope)) >= 2.0, 1.0, 0.0)

IndR_his_u_slope_ens = IndR_his_u_slope.mean(dim="models", skipna=True)
IndR_his_u_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_u_slope_ens, IndR_his_u_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_u_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_u_slope_ens = IndR_ssp585_p3_u_slope.mean(dim="models", skipna=True)
IndR_ssp585_p3_u_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_u_slope_ens, IndR_ssp585_p3_u_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_u_slope)) >= 2.0, 1.0, 0.0)

IndR_his_v_slope_ens = IndR_his_v_slope.mean(dim="models", skipna=True)
IndR_his_v_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_v_slope_ens, IndR_his_v_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_v_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_v_slope_ens = IndR_ssp585_p3_v_slope.mean(dim="models", skipna=True)
IndR_ssp585_p3_v_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_v_slope_ens, IndR_ssp585_p3_v_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_v_slope)) >= 2.0, 1.0, 0.0)

IndR_his_hgt_rvalue_ens = ca.cal_rMME(IndR_his_hgt_rvalue, "models")
IndR_his_hgt_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_hgt_rvalue_ens, IndR_his_hgt_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_hgt_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_hgt_rvalue_ens = ca.cal_rMME(IndR_ssp585_p3_hgt_rvalue, "models")
IndR_ssp585_p3_hgt_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_hgt_rvalue_ens, IndR_ssp585_p3_hgt_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_hgt_slope)) >= 2.0, 1.0, 0.0)

IndR_his_u_rvalue_ens = ca.cal_rMME(IndR_his_u_rvalue, "models")
IndR_his_u_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_u_rvalue_ens, IndR_his_u_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_u_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_u_rvalue_ens = ca.cal_rMME(IndR_ssp585_p3_u_rvalue, "models")
IndR_ssp585_p3_u_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_u_rvalue_ens, IndR_ssp585_p3_u_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_u_slope)) >= 2.0, 1.0, 0.0)

IndR_his_v_rvalue_ens = ca.cal_rMME(IndR_his_v_rvalue, "models")
IndR_his_v_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_v_rvalue_ens, IndR_his_v_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_v_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_v_rvalue_ens = ca.cal_rMME(IndR_ssp585_p3_v_rvalue, "models")
IndR_ssp585_p3_v_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_v_rvalue_ens, IndR_ssp585_p3_v_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_v_slope)) >= 2.0, 1.0, 0.0)

IndR_his_wind_ens_mask = ca.wind_check(
    xr.where(IndR_his_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_ens_mask > 0.0, 1.0, 0.0),
)
IndR_ssp585_p3_wind_ens_mask = ca.wind_check(
    xr.where(IndR_ssp585_p3_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_u_slope_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_ens_mask > 0.0, 1.0, 0.0),
)
# %%
#   plot the rvalue of hgt&u&v regress onto IndR in ERA5 and historical
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 31), (6, 5))
    plot_array[-1,-1] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndRCRU_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRCRU_ERA5_hgt_rvalue.sel(level=lev), axs[0], n, np.where(IndRCRU_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRCRU_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndRCRU_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
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
        IndRCRU_ERA5_u_rvalue.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRCRU_ERA5_v_rvalue.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="CRU & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndRGPCP_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRGPCP_ERA5_hgt_rvalue.sel(level=lev), axs[1], n, np.where(IndRGPCP_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndRGPCP_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndRGPCP_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
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
        IndRGPCP_ERA5_u_rvalue.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRGPCP_ERA5_v_rvalue.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="GPCP & ERA5",
    )
    # ======================================
    con = axs[2].contourf(
        IndR_his_hgt_rvalue_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_rvalue_ens.sel(level=lev), axs[2], n, np.where(IndR_his_hgt_rvalue_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[2].quiver(
        IndR_his_u_rvalue_ens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_rvalue_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[2].quiver(
        IndR_his_u_rvalue_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_rvalue_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[2].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+3].contourf(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+3], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+3].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+3].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+3].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+3].format(
            rtitle="1979-2014", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   calculate the ranking of different models
lat = prehis_JJA.coords["lat"]
lon = prehis_JJA.coords["lon"]
lat_ranking_range = lat[(lat>=15) & (lat<=47.5)]
lon_ranking_range = lon[(lon>=50) & (lon<=140.0)]

IndR_ranking_list = []
IndR_hgt_pcc = []
IndR_u_pcc = []
IndR_v_pcc = []

IndR_hgt_RMSE = []
IndR_hgt_std = []
IndR_u_RMSE = []
IndR_u_std = []
IndR_v_RMSE = []
IndR_v_std = []

for num_mod, mod in enumerate(models):
    hgt_pcc = ca.cal_pcc(IndRGPCP_ERA5_hgt_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0), IndR_his_hgt_slope.sel(models=mod, lat=lat_ranking_range, lon=lon_ranking_range, level=200.0))
    u_pcc = ca.cal_pcc(IndRGPCP_ERA5_u_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0), IndR_his_u_slope.sel(models=mod, lat=lat_ranking_range, lon=lon_ranking_range, level=200.0))
    v_pcc = ca.cal_pcc(IndRGPCP_ERA5_v_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0), IndR_his_v_slope.sel(models=mod, lat=lat_ranking_range, lon=lon_ranking_range, level=200.0))
    
    IndR_ranking_list.append({"models": mod.data, "pcc": hgt_pcc+u_pcc+v_pcc})
    # IndR_ranking_list.append({"models":mod.data, "pcc":hgt_pcc})
    
    IndR_hgt_pcc.append(hgt_pcc)
    IndR_u_pcc.append(u_pcc)
    IndR_v_pcc.append(v_pcc)
    
    IndR_hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope.sel(models=mod,lat=lat_ranking_range,lon=lon_ranking_range, level=200.0)-IndRGPCP_ERA5_hgt_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
    IndR_u_RMSE.append(np.sqrt(np.power((IndR_his_u_slope.sel(models=mod,lat=lat_ranking_range,lon=lon_ranking_range, level=200.0)-IndRGPCP_ERA5_u_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
    IndR_v_RMSE.append(np.sqrt(np.power((IndR_his_v_slope.sel(models=mod,lat=lat_ranking_range,lon=lon_ranking_range, level=200.0)-IndRGPCP_ERA5_v_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
    
    IndR_hgt_std.append(float((IndR_his_hgt_slope.sel(models=mod,lat=lat_ranking_range,lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRGPCP_ERA5_hgt_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
    IndR_u_std.append(float((IndR_his_u_slope.sel(models=mod,lat=lat_ranking_range,lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRGPCP_ERA5_u_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
    IndR_v_std.append(float((IndR_his_v_slope.sel(models=mod,lat=lat_ranking_range,lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRGPCP_ERA5_v_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
#   for MME

IndR_hgt_pcc.append(ca.cal_pcc(IndRGPCP_ERA5_hgt_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0), IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)))
IndR_u_pcc.append(ca.cal_pcc(IndRGPCP_ERA5_u_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0), IndR_his_u_slope_ens.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)))
IndR_v_pcc.append(ca.cal_pcc(IndRGPCP_ERA5_v_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0), IndR_his_v_slope_ens.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)))

IndR_hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range,lon=lon_ranking_range, level=200.0)-IndRGPCP_ERA5_hgt_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
IndR_u_RMSE.append(np.sqrt(np.power((IndR_his_u_slope_ens.sel(lat=lat_ranking_range,lon=lon_ranking_range, level=200.0)-IndRGPCP_ERA5_u_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
IndR_v_RMSE.append(np.sqrt(np.power((IndR_his_v_slope_ens.sel(lat=lat_ranking_range,lon=lon_ranking_range, level=200.0)-IndRGPCP_ERA5_v_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))

IndR_hgt_std.append(float((IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range,lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRGPCP_ERA5_hgt_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
IndR_u_std.append(float((IndR_his_u_slope_ens.sel(lat=lat_ranking_range,lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRGPCP_ERA5_u_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
IndR_v_std.append(float((IndR_his_v_slope_ens.sel(lat=lat_ranking_range,lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRGPCP_ERA5_v_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)).data))

#   pick up the good models and calculate the gMME for hgt, u, v, precip
gmodels = ["CESM2-WACCM", "CMCC-ESM2","CAMS-CSM1-0", "INM-CM4-8", "MIROC-ES2L", "UKESM1-0-LL"]

pre_his_India_pre_slope_gens = pre_his_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)
pre_ssp585_p3_India_pre_slope_gens = pre_ssp585_p3_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)

pre_his_India_pre_slope_gens_mask = ca.MME_reg_mask(pre_his_India_pre_slope_gens, pre_his_India_pre_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)
pre_ssp585_p3_India_pre_slope_gens_mask = ca.MME_reg_mask(pre_ssp585_p3_India_pre_slope_gens, pre_ssp585_p3_India_pre_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)


IndR_his_hgt_slope_gens = IndR_his_hgt_slope.sel(models=gmodels).mean(dim="models", skipna=True)
IndR_his_u_slope_gens = IndR_his_u_slope.sel(models=gmodels).mean(dim="models", skipna=True)
IndR_his_v_slope_gens = IndR_his_v_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_ssp585_p3_hgt_slope_gens = IndR_ssp585_p3_hgt_slope.sel(models=gmodels).mean(dim="models", skipna=True)
IndR_ssp585_p3_u_slope_gens = IndR_ssp585_p3_u_slope.sel(models=gmodels).mean(dim="models", skipna=True)
IndR_ssp585_p3_v_slope_gens = IndR_ssp585_p3_v_slope.sel(models=gmodels).mean(dim="models", skipna=True)


pre_his_India_pre_rvalue_gens = ca.cal_rMME(pre_his_India_pre_rvalue.sel(models=gmodels), "models")
pre_ssp585_p3_India_pre_rvalue_gens = ca.cal_rMME(pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels), "models")

IndR_his_hgt_rvalue_gens = ca.cal_rMME(IndR_his_hgt_rvalue.sel(models=gmodels), "models")
IndR_his_u_rvalue_gens = ca.cal_rMME(IndR_his_u_rvalue.sel(models=gmodels), "models")
IndR_his_v_rvalue_gens = ca.cal_rMME(IndR_his_v_rvalue.sel(models=gmodels), "models")

IndR_ssp585_p3_hgt_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_hgt_rvalue.sel(models=gmodels), "models")
IndR_ssp585_p3_u_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_u_rvalue.sel(models=gmodels), "models")
IndR_ssp585_p3_v_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_v_rvalue.sel(models=gmodels), "models")

IndR_his_hgt_slope_gens_mask = ca.MME_reg_mask(IndR_his_hgt_slope_gens, IndR_his_hgt_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)
IndR_his_u_slope_gens_mask = ca.MME_reg_mask(IndR_his_u_slope_gens, IndR_his_u_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)
IndR_his_v_slope_gens_mask = ca.MME_reg_mask(IndR_his_v_slope_gens, IndR_his_v_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)

IndR_ssp585_p3_hgt_slope_gens_mask = ca.MME_reg_mask(IndR_ssp585_p3_hgt_slope_gens, IndR_ssp585_p3_hgt_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)
IndR_ssp585_p3_u_slope_gens_mask = ca.MME_reg_mask(IndR_ssp585_p3_u_slope_gens, IndR_ssp585_p3_u_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)
IndR_ssp585_p3_v_slope_gens_mask = ca.MME_reg_mask(IndR_ssp585_p3_v_slope_gens, IndR_ssp585_p3_v_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)

pre_his_India_pre_rvalue_gens_mask = ca.MME_reg_mask(pre_his_India_pre_rvalue_gens, pre_his_India_pre_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)
pre_ssp585_p3_India_pre_rvalue_gens_mask = ca.MME_reg_mask(pre_ssp585_p3_India_pre_rvalue_gens, pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)

IndR_his_hgt_rvalue_gens_mask = ca.MME_reg_mask(IndR_his_hgt_rvalue_gens, IndR_his_hgt_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)
IndR_his_u_rvalue_gens_mask = ca.MME_reg_mask(IndR_his_u_rvalue_gens, IndR_his_u_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)
IndR_his_v_rvalue_gens_mask = ca.MME_reg_mask(IndR_his_v_rvalue_gens, IndR_his_v_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)

IndR_ssp585_p3_hgt_rvalue_gens_mask = ca.MME_reg_mask(IndR_ssp585_p3_hgt_rvalue_gens, IndR_ssp585_p3_hgt_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)
IndR_ssp585_p3_u_rvalue_gens_mask = ca.MME_reg_mask(IndR_ssp585_p3_u_rvalue_gens, IndR_ssp585_p3_u_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)
IndR_ssp585_p3_v_rvalue_gens_mask = ca.MME_reg_mask(IndR_ssp585_p3_v_rvalue_gens, IndR_ssp585_p3_v_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True)

IndR_his_wind_gens_mask = ca.wind_check(
    xr.where(IndR_his_u_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_u_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_gens_mask > 0.0, 1.0, 0.0),
)

IndR_ssp585_p3_wind_gens_mask = ca.wind_check(
    xr.where(IndR_ssp585_p3_u_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_u_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_gens_mask > 0.0, 1.0, 0.0),
)
#   for good models MME

IndR_hgt_pcc.append(ca.cal_pcc(IndRGPCP_ERA5_hgt_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0), IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)))
IndR_u_pcc.append(ca.cal_pcc(IndRGPCP_ERA5_u_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0), IndR_his_u_slope_gens.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)))
IndR_v_pcc.append(ca.cal_pcc(IndRGPCP_ERA5_v_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0), IndR_his_v_slope_gens.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)))

IndR_hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range,lon=lon_ranking_range, level=200.0)-IndRGPCP_ERA5_hgt_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
IndR_u_RMSE.append(np.sqrt(np.power((IndR_his_u_slope_gens.sel(lat=lat_ranking_range,lon=lon_ranking_range, level=200.0)-IndRGPCP_ERA5_u_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
IndR_v_RMSE.append(np.sqrt(np.power((IndR_his_v_slope_gens.sel(lat=lat_ranking_range,lon=lon_ranking_range, level=200.0)-IndRGPCP_ERA5_v_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))

IndR_hgt_std.append(float((IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range,lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRGPCP_ERA5_hgt_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
IndR_u_std.append(float((IndR_his_u_slope_gens.sel(lat=lat_ranking_range,lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRGPCP_ERA5_u_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
IndR_v_std.append(float((IndR_his_v_slope_gens.sel(lat=lat_ranking_range,lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRGPCP_ERA5_v_slope.sel(lat=lat_ranking_range, lon=lon_ranking_range, level=200.0).std(dim=["lat","lon"],skipna=True)).data))

print(sorted(IndR_ranking_list, key=lambda x : x["pcc"]))
# %%
#   plot the taylor-diagram
labels = list(models.data)
labels.append("MME")
labels.append("gMME")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#绘图
fig=plt.figure(figsize=(12,8),dpi=300)
plt.rc('font',family='Arial',size=13)

#调用函数
ax1=fig.add_subplot(111,projection='polar')
box = ax1.get_position()
ax1.set_position([0, box.y0, box.width*1.2, box.height])
# ax1.text(0.6,0.1,'(a)',fontsize=15)
# tar(ax1,np.array(IndR_EAM_pcc),np.array(IndR_EAM_std),labels)
sepl.taylor_diagram(ax1,np.array(IndR_hgt_pcc),np.array(IndR_hgt_std), dotlables=labels, lables=True, color="r")
sepl.taylor_diagram(ax1,np.array(IndR_u_pcc),np.array(IndR_u_std), color="b")
sepl.taylor_diagram(ax1,np.array(IndR_v_pcc),np.array(IndR_v_std), color="g")
plt.legend(loc="center left", bbox_to_anchor=(1.1,0.5), ncol=2, frameon=True, numpoints=1, handlelength=0)

# %%
#   plot the 200hPa good-models for corr coeff.
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 13), (3, 4))
    plot_array[-1,-3:] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndRCRU_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndRCRU_ERA5_hgt_rvalue.sel(level=lev), axs[0], n, np.where(IndRCRU_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRCRU_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndRCRU_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
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
        IndRCRU_ERA5_u_rvalue.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRCRU_ERA5_v_rvalue.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="CRU & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndRGPCP_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndRGPCP_ERA5_hgt_rvalue.sel(level=lev), axs[1], n, np.where(IndRGPCP_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndRGPCP_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndRGPCP_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
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
        IndRGPCP_ERA5_u_rvalue.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRGPCP_ERA5_v_rvalue.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="GPCP & ERA5",
    )
    # ======================================
    con = axs[2].contourf(
        IndR_his_hgt_rvalue_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_his_hgt_rvalue_gens.sel(level=lev), axs[2], n, np.where(IndR_his_hgt_rvalue_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[2].quiver(
        IndR_his_u_rvalue_gens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_rvalue_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[2].quiver(
        IndR_his_u_rvalue_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_rvalue_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[2].format(
        rtitle="1979-2014", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+3].contourf(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+3], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+3].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+3].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+3].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+3].format(
            rtitle="1979-2014", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the 200hPa good-models for reg coeff.
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 13), (3, 4))
    plot_array[-1,-3:] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndRCRU_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRCRU_ERA5_hgt_slope.sel(level=lev), axs[0], n, np.where(IndRCRU_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRCRU_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRCRU_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
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
        IndRCRU_ERA5_u_slope.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRCRU_ERA5_v_slope.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="CRU & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndRGPCP_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRGPCP_ERA5_hgt_slope.sel(level=lev), axs[1], n, np.where(IndRGPCP_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndRGPCP_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRGPCP_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
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
        IndRGPCP_ERA5_u_slope.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRGPCP_ERA5_v_slope.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="GPCP & ERA5",
    )
    # ======================================
    con = axs[2].contourf(
        IndR_his_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_gens.sel(level=lev), axs[2], n, np.where(IndR_his_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[2].quiver(
        IndR_his_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_gens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[2].quiver(
        IndR_his_u_slope_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_gens.sel(level=lev).where(IndR_his_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[2].format(
        rtitle="1979-2014", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+3].contourf(
            IndR_his_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_his_hgt_slope.sel(models=mod,level=lev), axs[num_mod+3], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+3].quiver(
            IndR_his_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+3].quiver(
            IndR_his_u_slope.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_slope.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+3].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+3].format(
            rtitle="1979-2014", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the precipitation fields of good-models for corr coeff.
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 13), (3, 4))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
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
# ===================================================
con = axs[0].contourf(
    pre_CRU_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_CRU_India_pre_rvalue, axs[0], n, np.where(pre_CRU_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="CRU",
)
# ===================================================
con = axs[1].contourf(
    pre_GPCP_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_GPCP_India_pre_rvalue, axs[1], n, np.where(pre_GPCP_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="GPCP",
)
# # ===================================================
con = axs[2].contourf(
    pre_his_India_pre_rvalue_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_his_India_pre_rvalue_gens, axs[2], n, np.where(pre_his_India_pre_rvalue_gens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[2].format(
    rtitle="1979-2014", ltitle="gMME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+3].contourf(
    pre_his_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_his_India_pre_rvalue.sel(models=mod), axs[num_models+3], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.10), "bright purple", 4.0,
    )

    axs[num_models+3].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the precipitation fields of good-models for reg coeff.
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 13), (3, 4))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
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
# ===================================================
con = axs[0].contourf(
    pre_CRU_India_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_CRU_India_pre_slope, axs[0], n, np.where(pre_CRU_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="CRU",
)
# ===================================================
con = axs[1].contourf(
    pre_GPCP_India_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_GPCP_India_pre_slope, axs[1], n, np.where(pre_GPCP_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="GPCP",
)
# # ===================================================
con = axs[2].contourf(
    pre_his_India_pre_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_his_India_pre_slope_gens, axs[2], n, np.where(pre_his_India_pre_slope_gens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[2].format(
    rtitle="1979-2014", ltitle="gMME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+3].contourf(
    pre_his_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both"
    )
    sepl.plt_sig(
        pre_his_India_pre_slope.sel(models=mod), axs[num_models+3], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.10), "bright purple", 4.0,
    )

    axs[num_models+3].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the 200hPa circulation in good-models for corr coeff. in ssp585_p3
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 10), (3, 3))
    plot_array[-1,-2:] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_ssp585_p3_hgt_rvalue_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_rvalue_gens.sel(level=lev), axs[0], n, np.where(IndR_ssp585_p3_hgt_rvalue_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndR_ssp585_p3_u_rvalue_gens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_p3_v_rvalue_gens.sel(level=lev)[::ski, ::ski],
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
        IndR_ssp585_p3_u_rvalue_gens.sel(level=lev).where(IndR_ssp585_p3_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_p3_v_rvalue_gens.sel(level=lev).where(IndR_ssp585_p3_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="2064-2099", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+1].contourf(
            IndR_ssp585_p3_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_ssp585_p3_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IndR_ssp585_p3_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_ssp585_p3_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_rvalue.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_ssp585_p3_v_rvalue.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="2064-2099", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the 200hPa circulation in good-models for reg coeff. in ssp585_p3
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 10), (3, 3))
    plot_array[-1,-2:] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_ssp585_p3_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_slope_gens.sel(level=lev), axs[0], n, np.where(IndR_ssp585_p3_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndR_ssp585_p3_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_gens.sel(level=lev)[::ski, ::ski],
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
        IndR_ssp585_p3_u_slope_gens.sel(level=lev).where(IndR_ssp585_p3_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_gens.sel(level=lev).where(IndR_ssp585_p3_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="2064-2099", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+1].contourf(
            IndR_ssp585_p3_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_ssp585_p3_hgt_slope.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IndR_ssp585_p3_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_ssp585_p3_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_slope.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_ssp585_p3_v_slope.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="2064-2099", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the precipitation fields of good-models for corr coeff. in ssp585_p3
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 10), (3, 3))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
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
# ===================================================
con = axs[0].contourf(
    pre_ssp585_p3_India_pre_rvalue_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_ssp585_p3_India_pre_rvalue_gens, axs[0], n, np.where(pre_ssp585_p3_India_pre_rvalue_gens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[0].format(
    rtitle="2064-2099", ltitle="gMME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_ssp585_p3_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_ssp585_p3_India_pre_rvalue.sel(models=mod), axs[num_models+1], n, np.where(pre_ssp585_p3_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.10), "bright purple", 4.0,
    )

    axs[num_models+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the precipitation fields of good-models for reg coeff. in ssp585_p3
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 10), (3, 3))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
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
# ===================================================
con = axs[0].contourf(
    pre_ssp585_p3_India_pre_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0,2.1, 0.1),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_ssp585_p3_India_pre_slope_gens, axs[0], n, np.where(pre_ssp585_p3_India_pre_slope_gens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[0].format(
    rtitle="2064-2099", ltitle="gMME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_ssp585_p3_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0,2.1, 0.1),
    zorder=0.8,
    extend="both"
    )
    sepl.plt_sig(
        pre_ssp585_p3_India_pre_slope.sel(models=mod), axs[num_models+1], n, np.where(pre_ssp585_p3_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.10), "bright purple", 4.0,
    )

    axs[num_models+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   calculate the good models difference between historical run and ssp585_p3 run
pre_diff_India_pre_slope = pre_ssp585_p3_India_pre_slope - pre_his_India_pre_slope

pre_diff_India_pre_mask = ca.cal_mmemask(pre_diff_India_pre_slope)

pre_diff_India_pre_slope_gens = pre_diff_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)

pre_diff_India_pre_gens_mask = ca.cal_mmemask(pre_diff_India_pre_slope.sel(models=gmodels))

pre_diff_India_pre_rvalue = ca.cal_rdiff(pre_ssp585_p3_India_pre_rvalue, pre_his_India_pre_rvalue)
pre_diff_India_pre_rvalue_gens = ca.cal_rMME(pre_diff_India_pre_rvalue.sel(models=gmodels), "models")

IndR_diff_hgt_slope = IndR_ssp585_p3_hgt_slope - IndR_his_hgt_slope
IndR_diff_hgt_slope_gens = IndR_diff_hgt_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_hgt_mask = ca.cal_mmemask(IndR_diff_hgt_slope)
IndR_diff_hgt_gens_mask = ca.cal_mmemask(IndR_diff_hgt_slope.sel(models=gmodels))

IndR_diff_u_slope = IndR_ssp585_p3_u_slope - IndR_his_u_slope
IndR_diff_u_slope_gens = IndR_diff_u_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_u_mask = ca.cal_mmemask(IndR_diff_u_slope)
IndR_diff_u_gens_mask = ca.cal_mmemask(IndR_diff_u_slope.sel(models=gmodels))

IndR_diff_v_slope = IndR_ssp585_p3_v_slope - IndR_his_v_slope
IndR_diff_v_slope_gens = IndR_diff_v_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_v_mask = ca.cal_mmemask(IndR_diff_v_slope)
IndR_diff_v_gens_mask = ca.cal_mmemask(IndR_diff_v_slope.sel(models=gmodels))

IndR_diff_wind_mask = ca.wind_check(
    xr.where(IndR_diff_u_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_u_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_mask > 0.0, 1.0, 0.0),
)
IndR_diff_wind_gens_mask = ca.wind_check(
    xr.where(IndR_diff_u_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_u_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_gens_mask > 0.0, 1.0, 0.0),
)

IndR_diff_hgt_rvalue = ca.cal_rdiff(IndR_ssp585_p3_hgt_rvalue, IndR_his_hgt_rvalue)
IndR_diff_hgt_rvalue_gens = ca.cal_rMME(IndR_diff_hgt_rvalue.sel(models=gmodels), "models")

IndR_diff_u_rvalue = ca.cal_rdiff(IndR_ssp585_p3_u_rvalue, IndR_his_u_rvalue)
IndR_diff_u_rvalue_gens = ca.cal_rMME(IndR_diff_u_rvalue.sel(models=gmodels), "models")

IndR_diff_v_rvalue = ca.cal_rdiff(IndR_ssp585_p3_v_rvalue, IndR_his_v_rvalue)
IndR_diff_v_rvalue_gens = ca.cal_rMME(IndR_diff_v_rvalue.sel(models=gmodels), "models")
# %%
#   plot the corr coeff. precipitation difference between historical and ssp585
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 10), (3, 3))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
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
# ===================================================
con = axs[0].contourf(
    pre_diff_India_pre_rvalue_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )

axs[0].format(
    rtitle="diff", ltitle="gMME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_diff_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )

    axs[num_models+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the reg coeff. precipitation difference between historical and ssp585
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 10), (3, 3))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
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
# ===================================================
con = axs[0].contourf(
    pre_diff_India_pre_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both"
    )

axs[0].format(
    rtitle="diff", ltitle="gMME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+1].contourf(
    pre_diff_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both"
    )

    axs[num_models+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the 200hPa circulation in good-models for corr coeff. in diff
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 10), (3, 3))
    plot_array[-1,-2:] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_diff_hgt_rvalue_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_diff_hgt_rvalue_gens.sel(level=lev), axs[0], n, np.where(IndR_diff_hgt_gens_mask.sel(level=lev)[::n, ::n] > 0), "bright purple", 4.0,
    )
    
    axs[0].quiver(
        IndR_diff_u_rvalue_gens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_rvalue_gens.sel(level=lev)[::ski, ::ski],
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
        IndR_diff_u_rvalue_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev)>0)[::ski, ::ski],
        IndR_diff_v_rvalue_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev)>0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="diff", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+1].contourf(
            IndR_diff_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )

        m = axs[num_mod+1].quiver(
            IndR_diff_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_diff_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="diff", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the 200hPa circulation in good-models for reg coeff. in diff
startlevel=[-22, -15, -10]
spacinglevel=[1.1, 0.75, 0.5]
scalelevel=[0.14, 0.13, 0.13]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 10), (3, 3))
    plot_array[-1,-2:] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_diff_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.0},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_diff_hgt_rvalue_gens.sel(level=lev), axs[0], n, np.where(IndR_diff_hgt_gens_mask.sel(level=lev)[::n, ::n] > 0), "bright purple", 4.0,
    )

    axs[0].quiver(
        IndR_diff_u_slope_gens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_slope_gens.sel(level=lev)[::ski, ::ski],
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
        IndR_diff_u_slope_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev)>0)[::ski, ::ski],
        IndR_diff_v_slope_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev)>0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="diff", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+1].contourf(
            IndR_diff_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": 0.0},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )

        m = axs[num_mod+1].quiver(
            IndR_diff_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_diff_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="diff", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   calculate the corr analysis between India and different variables
#   NCR
IndR_GPCP_NC_regress = stats.linregress(preAIR_JJA.sel(time=preAIR_JJA.time.dt.year>=1979), preGPCP_NC_JJA)

IndR_his_NC_regress = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), prehis_NC_JJA.sel(time=prehis_NC_JJA.time.dt.year>=1979))

IndR_ssp585_p3_NC_regress = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), pressp585_NC_JJA.sel(time=pressp585_NC_JJA.time.dt.year>=2064))

IndR_diff_NC_slope = IndR_ssp585_p3_NC_regress[0] - IndR_his_NC_regress[0]
IndR_diff_NC_rvalue = ca.cal_rdiff(IndR_ssp585_p3_NC_regress[2], IndR_his_NC_regress[2])

#   KP
IndR_GPCP_KP_regress = stats.linregress(preAIR_JJA.sel(time=preAIR_JJA.time.dt.year>=1979), preGPCP_KP_JJA)

IndR_his_KP_regress = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), prehis_KP_JJA.sel(time=prehis_KP_JJA.time.dt.year>=1979))

IndR_ssp585_p3_KP_regress = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), pressp585_KP_JJA.sel(time=pressp585_KP_JJA.time.dt.year>=2064))

IndR_diff_KP_slope = IndR_ssp585_p3_KP_regress[0] - IndR_his_KP_regress[0]
IndR_diff_KP_rvalue = ca.cal_rdiff(IndR_ssp585_p3_KP_regress[2], IndR_his_KP_regress[2])

#   SJ (actually Southern Japan and Korean peninsula)
IndR_GPCP_SJ_regress = stats.linregress(preAIR_JJA.sel(time=preAIR_JJA.time.dt.year>=1979), preGPCP_SJ_JJA)

IndR_his_SJ_regress = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), prehis_SJ_JJA.sel(time=prehis_SJ_JJA.time.dt.year>=1979))

IndR_ssp585_p3_SJ_regress = ca.dim_linregress(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), pressp585_SJ_JJA.sel(time=pressp585_SJ_JJA.time.dt.year>=2064))

IndR_diff_SJ_slope = IndR_ssp585_p3_SJ_regress[0] - IndR_his_SJ_regress[0]
IndR_diff_SJ_rvalue = ca.cal_rdiff(IndR_ssp585_p3_SJ_regress[2], IndR_his_KP_regress[2])


# %%
#   plot the singular scatter-plot for good models for reg coeff.
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
# cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 8, left=0.5)
cycle = pplt.Cycle("Qual1", 7)

#   hist-GPCP
axs[0].scatter(0.75, IndR_GPCP_NC_regress[0],marker="s", labels="GPCP", legend_kw={"ncols":4}, color="blue5", legend="b")
#   hist-gMME
axs[0].scatter(0.75, IndR_his_NC_regress[0].sel(models=gmodels).mean(dim="models", skipna=True), marker="*", labels="gMME", legend_kw={"ncols":4}, cycle=cycle, legend="b", markersize=100)
#   hist-gmodels
for num_models, mod in enumerate(gmodels):
    axs[0].scatter(0.75, IndR_his_NC_regress[0].sel(models=mod), cycle=cycle, legend="b", legend_kw={"ncols":4}, labels=mod, marker=".", markersize=100)
    
#   ssp585_p3-gMME
axs[0].scatter(1.50, IndR_ssp585_p3_NC_regress[0].sel(models=gmodels).mean(dim="models", skipna=True), marker="*", cycle=cycle, markersize=100)

#   ssp585_p3-gmodels
for num_models, mod in enumerate(gmodels):
    axs[0].scatter(1.50, IndR_ssp585_p3_NC_regress[0].sel(models=mod), cycle=cycle, marker=".", markersize=100)

axs[0].format(xlocator=[0.75, 1.5], ylim=(-0.2,0.5), xlim=(0,2.25), xformatter=["historical", "ssp585_p3"], tickminor=False)
fig.format(suptitle="NC reg IndR")
# %%
#   plot the singular scatter-plot for good models for reg coeff.
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
# cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 8, left=0.5)
cycle = pplt.Cycle("Qual1", 7)

#   hist-GPCP
axs[0].scatter(0.75, IndR_GPCP_NC_regress[2],marker="s", labels="GPCP", legend_kw={"ncols":4}, color="blue5", legend="b")
#   hist-gMME
axs[0].scatter(0.75, IndR_his_NC_regress[2].sel(models=gmodels).mean(dim="models", skipna=True), marker="*", labels="gMME", legend_kw={"ncols":4}, cycle=cycle, legend="b", markersize=100)
#   hist-gmodels
for num_models, mod in enumerate(gmodels):
    axs[0].scatter(0.75, IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend="b", legend_kw={"ncols":4}, labels=mod, marker=".", markersize=100)
    
#   ssp585_p3-gMME
axs[0].scatter(1.50, IndR_ssp585_p3_NC_regress[2].sel(models=gmodels).mean(dim="models", skipna=True), marker="*", cycle=cycle, markersize=100)

#   ssp585_p3-gmodels
for num_models, mod in enumerate(gmodels):
    axs[0].scatter(1.50, IndR_ssp585_p3_NC_regress[2].sel(models=mod), cycle=cycle, marker=".", markersize=100)

axs[0].format(xlocator=[0.75, 1.5], ylim=(-0.2,0.6), xlim=(0,2.25), xformatter=["historical", "ssp585_p3"], tickminor=False)
fig.format(suptitle="NC reg IndR")
# %%
#   plot the bar-plot of the IndR related NC precipitation for historical and ssp585_p3 (reg coeff.)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_NC_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_NC_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_NC_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_NC_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_NC_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_NC_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-0.7,0.7),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and NC")
# %%
#   plot the bar-plot of the IndR related NC precipitation for historical and ssp585_p3 (corr coeff.)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_NC_regress[2].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_NC_regress[2].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_NC_rvalue.sel(models=gmodels).data
plot_data[-1,0] = ca.cal_rMME(IndR_his_NC_regress[2].sel(models=gmodels), "models").data
plot_data[-1,1] = ca.cal_rMME(IndR_ssp585_p3_NC_regress[2].sel(models=gmodels), "models").data
plot_data[-1,2] = ca.cal_rMME(IndR_diff_NC_rvalue.sel(models=gmodels), "models").data

label_models = list(gmodels)
label_models.append("gMME")

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
m = axs[0].bar(label_models,plot_data,width=0.4,cycle="tab10",edgecolor="grey7")
axs[0].axhline(0,lw=1.5,color="grey7")
axs[0].axhline(ca.cal_rlim1(0.90, 36),lw=1.5,color="grey7",ls='--')
axs[0].axhline(-ca.cal_rlim1(0.90, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["historical", "ssp585_p3", "diff"])
axs[0].format(ylim=(-0.7,0.7),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Corr. Coeff. IndR and NC")
# %%
#   calculate the three models that have decrease IndR-NC trend in ssp585_p3
gmodels1 = ["CAMS-CSM1-0", "MIROC-ES2L", "UKESM1-0-LL"]

pre_his_India_pre_slope_g1ens = pre_his_India_pre_slope.sel(models=gmodels1).mean(dim="models", skipna=True)
pre_ssp585_p3_India_pre_slope_g1ens = pre_ssp585_p3_India_pre_slope.sel(models=gmodels1).mean(dim="models", skipna=True)

pre_his_India_pre_slope_g1ens_mask = ca.MME_reg_mask(pre_his_India_pre_slope_g1ens, pre_his_India_pre_slope.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)
pre_ssp585_p3_India_pre_slope_g1ens_mask = ca.MME_reg_mask(pre_ssp585_p3_India_pre_slope_g1ens, pre_ssp585_p3_India_pre_slope.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)


IndR_his_hgt_slope_g1ens = IndR_his_hgt_slope.sel(models=gmodels1).mean(dim="models", skipna=True)
IndR_his_u_slope_g1ens = IndR_his_u_slope.sel(models=gmodels1).mean(dim="models", skipna=True)
IndR_his_v_slope_g1ens = IndR_his_v_slope.sel(models=gmodels1).mean(dim="models", skipna=True)

IndR_ssp585_p3_hgt_slope_g1ens = IndR_ssp585_p3_hgt_slope.sel(models=gmodels1).mean(dim="models", skipna=True)
IndR_ssp585_p3_u_slope_g1ens = IndR_ssp585_p3_u_slope.sel(models=gmodels1).mean(dim="models", skipna=True)
IndR_ssp585_p3_v_slope_g1ens = IndR_ssp585_p3_v_slope.sel(models=gmodels1).mean(dim="models", skipna=True)


pre_his_India_pre_rvalue_g1ens = ca.cal_rMME(pre_his_India_pre_rvalue.sel(models=gmodels1), "models")
pre_ssp585_p3_India_pre_rvalue_g1ens = ca.cal_rMME(pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels1), "models")

IndR_his_hgt_rvalue_g1ens = ca.cal_rMME(IndR_his_hgt_rvalue.sel(models=gmodels1), "models")
IndR_his_u_rvalue_g1ens = ca.cal_rMME(IndR_his_u_rvalue.sel(models=gmodels1), "models")
IndR_his_v_rvalue_g1ens = ca.cal_rMME(IndR_his_v_rvalue.sel(models=gmodels1), "models")

IndR_ssp585_p3_hgt_rvalue_g1ens = ca.cal_rMME(IndR_ssp585_p3_hgt_rvalue.sel(models=gmodels1), "models")
IndR_ssp585_p3_u_rvalue_g1ens = ca.cal_rMME(IndR_ssp585_p3_u_rvalue.sel(models=gmodels1), "models")
IndR_ssp585_p3_v_rvalue_g1ens = ca.cal_rMME(IndR_ssp585_p3_v_rvalue.sel(models=gmodels1), "models")

IndR_his_hgt_slope_g1ens_mask = ca.MME_reg_mask(IndR_his_hgt_slope_g1ens, IndR_his_hgt_slope.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)
IndR_his_u_slope_g1ens_mask = ca.MME_reg_mask(IndR_his_u_slope_g1ens, IndR_his_u_slope.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)
IndR_his_v_slope_g1ens_mask = ca.MME_reg_mask(IndR_his_v_slope_g1ens, IndR_his_v_slope.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)

IndR_ssp585_p3_hgt_slope_g1ens_mask = ca.MME_reg_mask(IndR_ssp585_p3_hgt_slope_g1ens, IndR_ssp585_p3_hgt_slope.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)
IndR_ssp585_p3_u_slope_g1ens_mask = ca.MME_reg_mask(IndR_ssp585_p3_u_slope_g1ens, IndR_ssp585_p3_u_slope.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)
IndR_ssp585_p3_v_slope_g1ens_mask = ca.MME_reg_mask(IndR_ssp585_p3_v_slope_g1ens, IndR_ssp585_p3_v_slope.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)

pre_his_India_pre_rvalue_g1ens_mask = ca.MME_reg_mask(pre_his_India_pre_rvalue_g1ens, pre_his_India_pre_rvalue.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)
pre_ssp585_p3_India_pre_rvalue_g1ens_mask = ca.MME_reg_mask(pre_ssp585_p3_India_pre_rvalue_g1ens, pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)

IndR_his_hgt_rvalue_g1ens_mask = ca.MME_reg_mask(IndR_his_hgt_rvalue_g1ens, IndR_his_hgt_rvalue.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)
IndR_his_u_rvalue_g1ens_mask = ca.MME_reg_mask(IndR_his_u_rvalue_g1ens, IndR_his_u_rvalue.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)
IndR_his_v_rvalue_g1ens_mask = ca.MME_reg_mask(IndR_his_v_rvalue_g1ens, IndR_his_v_rvalue.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)

IndR_ssp585_p3_hgt_rvalue_g1ens_mask = ca.MME_reg_mask(IndR_ssp585_p3_hgt_rvalue_g1ens, IndR_ssp585_p3_hgt_rvalue.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)
IndR_ssp585_p3_u_rvalue_g1ens_mask = ca.MME_reg_mask(IndR_ssp585_p3_u_rvalue_g1ens, IndR_ssp585_p3_u_rvalue.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)
IndR_ssp585_p3_v_rvalue_g1ens_mask = ca.MME_reg_mask(IndR_ssp585_p3_v_rvalue_g1ens, IndR_ssp585_p3_v_rvalue.sel(models=gmodels1).std(dim="models", skipna=True), len(gmodels1), True)

IndR_his_wind_g1ens_mask = ca.wind_check(
    xr.where(IndR_his_u_slope_g1ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_g1ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_u_slope_g1ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_g1ens_mask > 0.0, 1.0, 0.0),
)

IndR_ssp585_p3_wind_g1ens_mask = ca.wind_check(
    xr.where(IndR_ssp585_p3_u_slope_g1ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_g1ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_u_slope_g1ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_g1ens_mask > 0.0, 1.0, 0.0),
)

pre_diff_India_pre_slope_g1ens = pre_diff_India_pre_slope.sel(models=gmodels1).mean(dim="models", skipna=True)

pre_diff_India_pre_rvalue_g1ens = ca.cal_rMME(pre_diff_India_pre_rvalue.sel(models=gmodels1), "models")

IndR_diff_hgt_slope_g1ens = IndR_diff_hgt_slope.sel(models=gmodels1).mean(dim="models", skipna=True)

IndR_diff_u_slope_g1ens = IndR_diff_u_slope.sel(models=gmodels1).mean(dim="models", skipna=True)

IndR_diff_v_slope_g1ens = IndR_diff_v_slope.sel(models=gmodels1).mean(dim="models", skipna=True)

IndR_diff_hgt_rvalue_g1ens = ca.cal_rMME(IndR_diff_hgt_rvalue.sel(models=gmodels1), "models")

IndR_diff_u_rvalue_g1ens = ca.cal_rMME(IndR_diff_u_rvalue.sel(models=gmodels1), "models")

IndR_diff_v_rvalue_g1ens = ca.cal_rMME(IndR_diff_v_rvalue.sel(models=gmodels1), "models")


# %%
#   plot the 200hPa good-models1 for corr coeff.
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 7), (3, 2))
    # plot_array[-1,-3:] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndRCRU_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndRCRU_ERA5_hgt_rvalue.sel(level=lev), axs[0], n, np.where(IndRCRU_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRCRU_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndRCRU_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
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
        IndRCRU_ERA5_u_rvalue.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRCRU_ERA5_v_rvalue.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="CRU & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndRGPCP_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndRGPCP_ERA5_hgt_rvalue.sel(level=lev), axs[1], n, np.where(IndRGPCP_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndRGPCP_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndRGPCP_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
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
        IndRGPCP_ERA5_u_rvalue.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRGPCP_ERA5_v_rvalue.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="GPCP & ERA5",
    )
    # ======================================
    con = axs[2].contourf(
        IndR_his_hgt_rvalue_g1ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_his_hgt_rvalue_g1ens.sel(level=lev), axs[2], n, np.where(IndR_his_hgt_rvalue_g1ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[2].quiver(
        IndR_his_u_rvalue_g1ens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_rvalue_g1ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[2].quiver(
        IndR_his_u_rvalue_g1ens.sel(level=lev).where(IndR_his_wind_g1ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_rvalue_g1ens.sel(level=lev).where(IndR_his_wind_g1ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[2].format(
        rtitle="1979-2014", ltitle="gMME1",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels1):
        con = axs[num_mod+3].contourf(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+3], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+3].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+3].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+3].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+3].format(
            rtitle="1979-2014", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the 200hPa good-models for reg coeff.
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 7), (3, 2))
    # plot_array[-1,-3:] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndRCRU_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRCRU_ERA5_hgt_slope.sel(level=lev), axs[0], n, np.where(IndRCRU_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRCRU_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRCRU_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
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
        IndRCRU_ERA5_u_slope.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRCRU_ERA5_v_slope.sel(level=lev).where(IndRCRU_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="CRU & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndRGPCP_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRGPCP_ERA5_hgt_slope.sel(level=lev), axs[1], n, np.where(IndRGPCP_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndRGPCP_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRGPCP_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
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
        IndRGPCP_ERA5_u_slope.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRGPCP_ERA5_v_slope.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="GPCP & ERA5",
    )
    # ======================================
    con = axs[2].contourf(
        IndR_his_hgt_slope_g1ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_g1ens.sel(level=lev), axs[2], n, np.where(IndR_his_hgt_slope_g1ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[2].quiver(
        IndR_his_u_slope_g1ens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_g1ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[2].quiver(
        IndR_his_u_slope_g1ens.sel(level=lev).where(IndR_his_wind_g1ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_g1ens.sel(level=lev).where(IndR_his_wind_g1ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[2].format(
        rtitle="1979-2014", ltitle="gMME1",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels1):
        con = axs[num_mod+3].contourf(
            IndR_his_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_his_hgt_slope.sel(models=mod,level=lev), axs[num_mod+3], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+3].quiver(
            IndR_his_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+3].quiver(
            IndR_his_u_slope.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_slope.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+3].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+3].format(
            rtitle="1979-2014", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the 200hPa circulation in good-models1 for corr coeff. in ssp585_p3
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 5), (2, 2))
    # plot_array[-1,-2:] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_ssp585_p3_hgt_rvalue_g1ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_rvalue_g1ens.sel(level=lev), axs[0], n, np.where(IndR_ssp585_p3_hgt_rvalue_g1ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndR_ssp585_p3_u_rvalue_g1ens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_p3_v_rvalue_g1ens.sel(level=lev)[::ski, ::ski],
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
        IndR_ssp585_p3_u_rvalue_g1ens.sel(level=lev).where(IndR_ssp585_p3_wind_g1ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_p3_v_rvalue_g1ens.sel(level=lev).where(IndR_ssp585_p3_wind_g1ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="2064-2099", ltitle="gMME1",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels1):
        con = axs[num_mod+1].contourf(
            IndR_ssp585_p3_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_ssp585_p3_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IndR_ssp585_p3_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_ssp585_p3_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_rvalue.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_ssp585_p3_v_rvalue.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="2064-2099", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the 200hPa circulation in good-models for reg coeff. in ssp585_p3
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 5), (2, 2))
    # plot_array[-1,-2:] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_ssp585_p3_hgt_slope_g1ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_slope_g1ens.sel(level=lev), axs[0], n, np.where(IndR_ssp585_p3_hgt_slope_g1ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndR_ssp585_p3_u_slope_g1ens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_g1ens.sel(level=lev)[::ski, ::ski],
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
        IndR_ssp585_p3_u_slope_g1ens.sel(level=lev).where(IndR_ssp585_p3_wind_g1ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_g1ens.sel(level=lev).where(IndR_ssp585_p3_wind_g1ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="2064-2099", ltitle="gMME1",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels1):
        con = axs[num_mod+1].contourf(
            IndR_ssp585_p3_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_ssp585_p3_hgt_slope.sel(models=mod,level=lev), axs[num_mod+1], n, np.where(IndR_ssp585_p3_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_ssp585_p3_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+1].quiver(
            IndR_ssp585_p3_u_slope.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_ssp585_p3_v_slope.sel(models=mod,level=lev).where(IndR_ssp585_p3_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="2064-2099", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the 200hPa circulation in good-models1 for corr coeff. in diff
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 5), (2, 2))
    # plot_array[-1,-2:] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_diff_hgt_rvalue_g1ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )

    m = axs[0].quiver(
        IndR_diff_u_rvalue_g1ens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_rvalue_g1ens.sel(level=lev)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="diff", ltitle="gMME1",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels1):
        con = axs[num_mod+1].contourf(
            IndR_diff_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )

        m = axs[num_mod+1].quiver(
            IndR_diff_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_diff_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="diff", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the 200hPa circulation in good-models for reg coeff. in diff
startlevel=[-22, -15, -10]
spacinglevel=[1.1, 0.75, 0.5]
scalelevel=[0.14, 0.13, 0.13]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 5), (2, 2))
    # plot_array[-1,-2:] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndR_diff_hgt_slope_g1ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )

    m = axs[0].quiver(
        IndR_diff_u_slope_g1ens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_slope_g1ens.sel(level=lev)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="diff", ltitle="gMME1",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels1):
        con = axs[num_mod+1].contourf(
            IndR_diff_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )

        m = axs[num_mod+1].quiver(
            IndR_diff_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_diff_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+1].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+1].format(
            rtitle="diff", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the correlation scatter-plot, x:pcc, y:corr(IndR, NCR)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
m = axs[0].scatter(1.0, IndR_GPCP_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="GPCP", marker="s", color="blue5")
for num_models, mod in enumerate(models_array):
    m = axs[0].scatter((np.array(IndR_hgt_pcc)+np.array(IndR_u_pcc)+np.array(IndR_v_pcc))[num_models]/3.0, IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod)
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter((np.array(IndR_hgt_pcc)+np.array(IndR_u_pcc)+np.array(IndR_v_pcc))[26]/3.0, ca.cal_rMME(IndR_his_NC_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^")
m = axs[0].scatter((np.array(IndR_hgt_pcc)+np.array(IndR_u_pcc)+np.array(IndR_v_pcc))[27]/3.0, ca.cal_rMME(IndR_his_NC_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*")
# #   第一象限
# axs[0].text(0.4,0.5,s='{} ({:.1f}%)'.format(IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]>0) & (IndR_his_SC_regress[2]>0)).count().data, IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]>0) & (IndR_his_SC_regress[2]>0)).count().data/26*100))
# #   第二象限
# axs[0].text(-0.55,0.5,s='{} ({:.1f}%)'.format(IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]>0) & (IndR_his_SC_regress[2]<0)).count().data, IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]>0) & (IndR_his_SC_regress[2]<0)).count().data/26*100))
# #   第三象限
# axs[0].text(-0.55,-0.5,s='{} ({:.1f}%)'.format(IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]<0) & (IndR_his_SC_regress[2]<0)).count().data, IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]<0) & (IndR_his_SC_regress[2]<0)).count().data/26*100))
# #   第四象限
# axs[0].text(0.4,-0.5,s='{} ({:.1f}%)'.format(IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]<0) & (IndR_his_SC_regress[2]>0)).count().data, IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]<0) & (IndR_his_SC_regress[2]>0)).count().data/26*100))
#   x-axis title
axs[0].text(-0.55,0.03,s='pcc_mean')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, NCR)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.2), ylim=(-0.6,0.6), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
#   plot the correlation scatter-plot, x:pcc, y:corr(IndR, NCR)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 8, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
m = axs[0].scatter(1.0, IndR_GPCP_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="GPCP", marker="s", color="blue5")
for (num_models, mod), num_pcc in zip(enumerate(gmodels), [list(models_array).index(gmod) for gmod in gmodels]):
    m = axs[0].scatter((np.array(IndR_hgt_pcc)+np.array(IndR_u_pcc)+np.array(IndR_v_pcc))[num_pcc]/3.0, IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod)
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# m = axs[0].scatter((IndR_hgt_pcc+IndR_u_pcc+IndR_v_pcc)[26]/3.0, ca.cal_rMME(IndR_his_NC_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^")
m = axs[0].scatter((np.array(IndR_hgt_pcc)+np.array(IndR_u_pcc)+np.array(IndR_v_pcc))[27]/3.0, ca.cal_rMME(IndR_his_NC_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*")
# #   第一象限
# axs[0].text(0.4,0.5,s='{} ({:.1f}%)'.format(IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]>0) & (IndR_his_SC_regress[2]>0)).count().data, IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]>0) & (IndR_his_SC_regress[2]>0)).count().data/26*100))
# #   第二象限
# axs[0].text(-0.55,0.5,s='{} ({:.1f}%)'.format(IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]>0) & (IndR_his_SC_regress[2]<0)).count().data, IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]>0) & (IndR_his_SC_regress[2]<0)).count().data/26*100))
# #   第三象限
# axs[0].text(-0.55,-0.5,s='{} ({:.1f}%)'.format(IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]<0) & (IndR_his_SC_regress[2]<0)).count().data, IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]<0) & (IndR_his_SC_regress[2]<0)).count().data/26*100))
# #   第四象限
# axs[0].text(0.4,-0.5,s='{} ({:.1f}%)'.format(IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]<0) & (IndR_his_SC_regress[2]>0)).count().data, IndR_his_NC_regress[2].where((IndR_his_NC_regress[2]<0) & (IndR_his_SC_regress[2]>0)).count().data/26*100))
#   x-axis title
axs[0].text(-0.55,0.03,s='pcc_mean')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, NCR)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.2), ylim=(-0.6,0.6), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
#   plot the rvalue of hgt&u&v regress onto IndR in ERA5 and historical, but for AIR data
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 31), (6, 5))
    plot_array[-1,-1] = 0
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
        # #   IWF area
        # x0 = 90
        # y0 = 5.0
        # width = 50.0
        # height = 27.5
        # patches(ax, x0 - cl, y0, width, height, proj)
    # ======================================
    con = axs[0].contourf(
        IndRAIR_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndRAIR_ERA5_hgt_rvalue.sel(level=lev), axs[0], n, np.where(IndRAIR_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRAIR_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndRAIR_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
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
        IndRAIR_ERA5_u_rvalue.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRAIR_ERA5_v_rvalue.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="AIR & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndRGPCP_ERA5_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndRGPCP_ERA5_hgt_rvalue.sel(level=lev), axs[1], n, np.where(IndRGPCP_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndRGPCP_ERA5_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndRGPCP_ERA5_v_rvalue.sel(level=lev)[::ski, ::ski],
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
        IndRGPCP_ERA5_u_rvalue.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRGPCP_ERA5_v_rvalue.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="GPCP & ERA5",
    )
    # ======================================
    con = axs[2].contourf(
        IndR_his_hgt_rvalue_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_his_hgt_rvalue_ens.sel(level=lev), axs[2], n, np.where(IndR_his_hgt_rvalue_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[2].quiver(
        IndR_his_u_rvalue_ens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_rvalue_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[2].quiver(
        IndR_his_u_rvalue_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_rvalue_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[2].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+3].contourf(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+3], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+3].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+3].quiver(
            IndR_his_u_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_rvalue.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+3].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+3].format(
            rtitle="1979-2014", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the correlation coefficients rvalue in AIR, GPCP, historical run
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (6, 5))
plot_array[5,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
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
# ===================================================
con = axs[0].contourf(
    pre_AIR_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_AIR_India_pre_rvalue, axs[0], n, np.where(pre_AIR_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="AIR",
)
# ===================================================
con = axs[1].contourf(
    pre_GPCP_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_GPCP_India_pre_rvalue, axs[1], n, np.where(pre_GPCP_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="GPCP",
)
# # ===================================================
con = axs[2].contourf(
    pre_his_India_pre_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_his_India_pre_rvalue_ens, axs[2], n, np.where(pre_his_India_pre_rvalue_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[2].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_his_India_pre_slope.coords["models"].data):
    con = axs[num_models+3].contourf(
    pre_his_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_his_India_pre_rvalue.sel(models=mod), axs[num_models+3], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.10), "bright purple", 4.0,
    )

    axs[num_models+3].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")

# %%
#   calculate the ranking of different models but use the AIR data as observation
lat = prehis_JJA.coords["lat"]
lon = prehis_JJA.coords["lon"]
#   for 200hPa
lat_ranking_range1 = lat[(lat>=15) & (lat<=47.5)]
lon_ranking_range1 = lon[(lon>=50) & (lon<=140.0)]

#   for 850hPa
lat_ranking_range2 = lat[(lat>=15) & (lat<=37.5)]
lon_ranking_range2 = lon[(lon>=110) & (lon<=137.0)]

IndR_ranking_list = []
IndR_200hgt_pcc = []

IndR_850hgt_pcc = []

IndR_200hgt_RMSE = []
IndR_200hgt_std = []

IndR_850hgt_RMSE = []
IndR_850hgt_std = []


for num_mod, mod in enumerate(models):
    
    hgt200_pcc = ca.cal_pcc(IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0), IndR_his_hgt_slope.sel(models=mod, lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0))
    
    hgt850_pcc = ca.cal_pcc(IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0), IndR_his_hgt_slope.sel(models=mod, lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0))
    
    # IndR_ranking_list.append({"models": mod.data, "pcc": hgt_pcc+u_pcc+v_pcc})
    IndR_ranking_list.append({"models":mod.data, "pcc":hgt200_pcc + hgt850_pcc})
    
    IndR_200hgt_pcc.append(hgt200_pcc)
    IndR_850hgt_pcc.append(hgt850_pcc)
    
    IndR_200hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope.sel(models=mod,lat=lat_ranking_range1,lon=lon_ranking_range1, level=200.0)-IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
    
    IndR_850hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope.sel(models=mod,lat=lat_ranking_range2,lon=lon_ranking_range2, level=850.0)-IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0)),2).mean(dim=["lat","lon"],skipna=True).data))
    
    IndR_200hgt_std.append(float((IndR_his_hgt_slope.sel(models=mod,lat=lat_ranking_range1,lon=lon_ranking_range1, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
    IndR_850hgt_std.append(float((IndR_his_hgt_slope.sel(models=mod,lat=lat_ranking_range2,lon=lon_ranking_range2, level=850.0).std(dim=["lat","lon"],skipna=True)/IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0).std(dim=["lat","lon"],skipna=True)).data))


#   for MME
IndR_200hgt_pcc.append(ca.cal_pcc(IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0), IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0)))
IndR_850hgt_pcc.append(ca.cal_pcc(IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0), IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0)))


IndR_200hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range1,lon=lon_ranking_range1, level=200.0)-IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
IndR_850hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range2,lon=lon_ranking_range2, level=850.0)-IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0)),2).mean(dim=["lat","lon"],skipna=True).data))



IndR_200hgt_std.append(float((IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range1,lon=lon_ranking_range1, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
IndR_850hgt_std.append(float((IndR_his_hgt_slope_ens.sel(lat=lat_ranking_range2,lon=lon_ranking_range2, level=850.0).std(dim=["lat","lon"],skipna=True)/IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0).std(dim=["lat","lon"],skipna=True)).data))


#   pick up the good models and calculate the gMME for hgt, u, v, precip
#   these gmodels are different from the ranking list calculated by the GPCP data
gmodels = ["CAMS-CSM1-0", "CESM2-WACCM", "CMCC-ESM2", "INM-CM4-8", "MRI-ESM2-0", "UKESM1-0-LL"]
# gmodels = ["CESM2-WACCM", "CMCC-ESM2", "MRI-ESM2-0", "UKESM1-0-LL"]

pre_his_India_pre_slope_gens = pre_his_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)
pre_ssp585_p3_India_pre_slope_gens = pre_ssp585_p3_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)

pre_his_India_pre_slope_gens_mask = xr.where((ca.MME_reg_mask(pre_his_India_pre_slope_gens, pre_his_India_pre_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(pre_his_India_pre_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
pre_ssp585_p3_India_pre_slope_gens_mask = xr.where((ca.MME_reg_mask(pre_ssp585_p3_India_pre_slope_gens, pre_ssp585_p3_India_pre_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(pre_ssp585_p3_India_pre_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)


IndR_his_hgt_slope_gens = IndR_his_hgt_slope.sel(models=gmodels).mean(dim="models", skipna=True)
IndR_his_u_slope_gens = IndR_his_u_slope.sel(models=gmodels).mean(dim="models", skipna=True)
IndR_his_v_slope_gens = IndR_his_v_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_ssp585_p3_hgt_slope_gens = IndR_ssp585_p3_hgt_slope.sel(models=gmodels).mean(dim="models", skipna=True)
IndR_ssp585_p3_u_slope_gens = IndR_ssp585_p3_u_slope.sel(models=gmodels).mean(dim="models", skipna=True)
IndR_ssp585_p3_v_slope_gens = IndR_ssp585_p3_v_slope.sel(models=gmodels).mean(dim="models", skipna=True)


pre_his_India_pre_rvalue_gens = ca.cal_rMME(pre_his_India_pre_rvalue.sel(models=gmodels), "models")
pre_ssp585_p3_India_pre_rvalue_gens = ca.cal_rMME(pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels), "models")

IndR_his_hgt_rvalue_gens = ca.cal_rMME(IndR_his_hgt_rvalue.sel(models=gmodels), "models")
IndR_his_u_rvalue_gens = ca.cal_rMME(IndR_his_u_rvalue.sel(models=gmodels), "models")
IndR_his_v_rvalue_gens = ca.cal_rMME(IndR_his_v_rvalue.sel(models=gmodels), "models")

IndR_ssp585_p3_hgt_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_hgt_rvalue.sel(models=gmodels), "models")
IndR_ssp585_p3_u_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_u_rvalue.sel(models=gmodels), "models")
IndR_ssp585_p3_v_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_v_rvalue.sel(models=gmodels), "models")

IndR_his_hgt_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_hgt_slope_gens, IndR_his_hgt_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_hgt_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_u_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_u_slope_gens, IndR_his_u_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_u_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_v_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_v_slope_gens, IndR_his_v_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_v_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_hgt_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_hgt_slope_gens, IndR_ssp585_p3_hgt_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_hgt_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_u_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_u_slope_gens, IndR_ssp585_p3_u_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_u_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_v_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_v_slope_gens, IndR_ssp585_p3_v_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_v_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

pre_his_India_pre_rvalue_gens_mask = xr.where((ca.MME_reg_mask(pre_his_India_pre_rvalue_gens, pre_his_India_pre_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(pre_his_India_pre_rvalue.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
pre_ssp585_p3_India_pre_rvalue_gens_mask = xr.where((ca.MME_reg_mask(pre_ssp585_p3_India_pre_rvalue_gens, pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

IndR_his_hgt_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_hgt_rvalue_gens, IndR_his_hgt_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_hgt_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_u_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_u_rvalue_gens, IndR_his_u_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_u_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_v_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_v_rvalue_gens, IndR_his_v_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_v_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_hgt_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_hgt_rvalue_gens, IndR_ssp585_p3_hgt_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_hgt_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_u_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_u_rvalue_gens, IndR_ssp585_p3_u_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_u_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_v_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_v_rvalue_gens, IndR_ssp585_p3_v_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_v_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

IndR_his_wind_gens_mask = ca.wind_check(
    xr.where(IndR_his_u_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_u_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_gens_mask > 0.0, 1.0, 0.0),
)

IndR_ssp585_p3_wind_gens_mask = ca.wind_check(
    xr.where(IndR_ssp585_p3_u_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_u_slope_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_gens_mask > 0.0, 1.0, 0.0),
)
#   for good models MME

IndR_200hgt_pcc.append(ca.cal_pcc(IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0), IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0)))
IndR_850hgt_pcc.append(ca.cal_pcc(IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0), IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0)))


IndR_200hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range1,lon=lon_ranking_range1, level=200.0)-IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0)),2).mean(dim=["lat","lon"],skipna=True).data))
IndR_850hgt_RMSE.append(np.sqrt(np.power((IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range2,lon=lon_ranking_range2, level=850.0)-IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0)),2).mean(dim=["lat","lon"],skipna=True).data))


IndR_200hgt_std.append(float((IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range1,lon=lon_ranking_range1, level=200.0).std(dim=["lat","lon"],skipna=True)/IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range1, lon=lon_ranking_range1, level=200.0).std(dim=["lat","lon"],skipna=True)).data))
IndR_850hgt_std.append(float((IndR_his_hgt_slope_gens.sel(lat=lat_ranking_range2,lon=lon_ranking_range2, level=850.0).std(dim=["lat","lon"],skipna=True)/IndRAIR_ERA5_hgt_slope.sel(lat=lat_ranking_range2, lon=lon_ranking_range2, level=850.0).std(dim=["lat","lon"],skipna=True)).data))


print(sorted(IndR_ranking_list, key=lambda x : x["pcc"]))
# %%
#   calculate the good models difference between historical run and ssp585_p3 run
pre_diff_India_pre_slope = pre_ssp585_p3_India_pre_slope - pre_his_India_pre_slope

pre_diff_India_pre_mask = ca.cal_mmemask(pre_diff_India_pre_slope)

pre_diff_India_pre_slope_gens = pre_diff_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)

pre_diff_India_pre_gens_mask = ca.cal_mmemask(pre_diff_India_pre_slope.sel(models=gmodels))

pre_diff_India_pre_rvalue = ca.cal_rdiff(pre_ssp585_p3_India_pre_rvalue, pre_his_India_pre_rvalue)
pre_diff_India_pre_rvalue_gens = ca.cal_rMME(pre_diff_India_pre_rvalue.sel(models=gmodels), "models")

IndR_diff_hgt_slope = IndR_ssp585_p3_hgt_slope - IndR_his_hgt_slope
IndR_diff_hgt_slope_gens = IndR_diff_hgt_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_hgt_mask = ca.cal_mmemask(IndR_diff_hgt_slope)
IndR_diff_hgt_gens_mask = ca.cal_mmemask(IndR_diff_hgt_slope.sel(models=gmodels))

IndR_diff_u_slope = IndR_ssp585_p3_u_slope - IndR_his_u_slope
IndR_diff_u_slope_gens = IndR_diff_u_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_u_mask = ca.cal_mmemask(IndR_diff_u_slope)
IndR_diff_u_gens_mask = ca.cal_mmemask(IndR_diff_u_slope.sel(models=gmodels))

IndR_diff_v_slope = IndR_ssp585_p3_v_slope - IndR_his_v_slope
IndR_diff_v_slope_gens = IndR_diff_v_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_v_mask = ca.cal_mmemask(IndR_diff_v_slope)
IndR_diff_v_gens_mask = ca.cal_mmemask(IndR_diff_v_slope.sel(models=gmodels))

IndR_diff_wind_mask = ca.wind_check(
    xr.where(IndR_diff_u_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_u_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_mask > 0.0, 1.0, 0.0),
)
IndR_diff_wind_gens_mask = ca.wind_check(
    xr.where(IndR_diff_u_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_u_gens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_gens_mask > 0.0, 1.0, 0.0),
)

IndR_diff_hgt_rvalue = ca.cal_rdiff(IndR_ssp585_p3_hgt_rvalue, IndR_his_hgt_rvalue)
IndR_diff_hgt_rvalue_gens = ca.cal_rMME(IndR_diff_hgt_rvalue.sel(models=gmodels), "models")

IndR_diff_u_rvalue = ca.cal_rdiff(IndR_ssp585_p3_u_rvalue, IndR_his_u_rvalue)
IndR_diff_u_rvalue_gens = ca.cal_rMME(IndR_diff_u_rvalue.sel(models=gmodels), "models")

IndR_diff_v_rvalue = ca.cal_rdiff(IndR_ssp585_p3_v_rvalue, IndR_his_v_rvalue)
IndR_diff_v_rvalue_gens = ca.cal_rMME(IndR_diff_v_rvalue.sel(models=gmodels), "models")
# %%
#   plot the taylor-diagram
labels = list(models.data)
labels.append("MME")
labels.append("gMME")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#绘图
fig=plt.figure(figsize=(12,8),dpi=300)
plt.rc('font',family='Arial',size=13)

#调用函数
ax1=fig.add_subplot(111,projection='polar')
box = ax1.get_position()
ax1.set_position([0, box.y0, box.width*1.2, box.height])
# ax1.text(0.6,0.1,'(a)',fontsize=15)
# tar(ax1,np.array(IndR_EAM_pcc),np.array(IndR_EAM_std),labels)
sepl.taylor_diagram(ax1,np.array(IndR_200hgt_pcc),np.array(IndR_200hgt_std), dotlables=labels, lables=True, color="r")
sepl.taylor_diagram(ax1,np.array(IndR_850hgt_pcc),np.array(IndR_850hgt_std), color="b")
plt.legend(loc="center left", bbox_to_anchor=(1.1,0.5), ncol=2, frameon=True, numpoints=1, handlelength=0)
# %%
#   plot the vorticity regress on the AIR
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
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
# ===================================================
con = axs[0].contourf(
    IndRAIR_ERA5_vor_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    IndRAIR_ERA5_vor_rvalue, axs[0], n, np.where(IndRAIR_ERA5_vor_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="AIR",
)
for num_mod, mod in enumerate(models):
    con = axs[num_mod+1].contourf(
        IndR_his_vor_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-1.0, 1.1, 0.1),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_his_vor_rvalue.sel(models=mod), axs[num_mod+1], n, np.where(IndR_his_vor_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+1].format(
        rtitle="1979-2014", ltitle="{}".format(mod.data),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="200hPa vor reg IndR")

# %%
#   plot the correlation scatter-plot, x:pcc, y:corr(IndR, KP)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
m = axs[0].scatter(1.0, IndR_GPCP_KP_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="GPCP", marker="s", color="blue5")
for num_models, mod in enumerate(models_array):
    m = axs[0].scatter((np.array(IndR_hgt_pcc)+np.array(IndR_u_pcc)+np.array(IndR_v_pcc))[num_models]/3.0, IndR_his_KP_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod)
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter((np.array(IndR_hgt_pcc)+np.array(IndR_u_pcc)+np.array(IndR_v_pcc))[26]/3.0, ca.cal_rMME(IndR_his_KP_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^")
m = axs[0].scatter((np.array(IndR_hgt_pcc)+np.array(IndR_u_pcc)+np.array(IndR_v_pcc))[27]/3.0, ca.cal_rMME(IndR_his_KP_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*")
# #   第一象限
# axs[0].text(0.4,0.5,s='{} ({:.1f}%)'.format(IndR_his_KP_regress[2].where((IndR_his_KP_regress[2]>0) & (IndR_his_SC_regress[2]>0)).count().data, IndR_his_KP_regress[2].where((IndR_his_KP_regress[2]>0) & (IndR_his_SC_regress[2]>0)).count().data/26*100))
# #   第二象限
# axs[0].text(-0.55,0.5,s='{} ({:.1f}%)'.format(IndR_his_KP_regress[2].where((IndR_his_KP_regress[2]>0) & (IndR_his_SC_regress[2]<0)).count().data, IndR_his_KP_regress[2].where((IndR_his_KP_regress[2]>0) & (IndR_his_SC_regress[2]<0)).count().data/26*100))
# #   第三象限
# axs[0].text(-0.55,-0.5,s='{} ({:.1f}%)'.format(IndR_his_KP_regress[2].where((IndR_his_KP_regress[2]<0) & (IndR_his_SC_regress[2]<0)).count().data, IndR_his_KP_regress[2].where((IndR_his_KP_regress[2]<0) & (IndR_his_SC_regress[2]<0)).count().data/26*100))
# #   第四象限
# axs[0].text(0.4,-0.5,s='{} ({:.1f}%)'.format(IndR_his_KP_regress[2].where((IndR_his_KP_regress[2]<0) & (IndR_his_SC_regress[2]>0)).count().data, IndR_his_KP_regress[2].where((IndR_his_KP_regress[2]<0) & (IndR_his_SC_regress[2]>0)).count().data/26*100))
#   x-axis title
axs[0].text(-0.55,0.03,s='pcc_mean')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, KPR)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.2), ylim=(-0.6,0.6), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
#   calculate the climatology relative vorticity in the historical and ssp585_p3
vorERA5_cli_ver_JJA = vorERA5_ver_JJA.sel(time=vorERA5_ver_JJA.time.dt.year>=1979).mean(dim="time", skipna=True)
vorhis_cli_ver_JJA = vorhis_ver_JJA.sel(time=vorhis_ver_JJA.time.dt.year>=1979).mean(dim="time", skipna=True)
vorssp585_p3_cli_ver_JJA = vorssp585_ver_JJA.sel(time=vorssp585_ver_JJA.time.dt.year>=2064).mean(dim="time", skipna=True)
# %%
#   plot the climatology relative vorticity in ERA5 and historical
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (6, 5))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
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
# ===================================================
con = axs[0].contourf(
    vorERA5_cli_ver_JJA,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-3.5e-5, 3.5e-5+3.5e-6, 3.5e-6),
    zorder=0.8,
    extend="both"
    )

axs[0].format(
    rtitle="1979-2014", ltitle="ERA5",
)
# ===================================================
con = axs[1].contourf(
    vorhis_cli_ver_JJA.mean(dim="models", skipna=True),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-3.5e-5, 3.5e-5+3.5e-6, 3.5e-6),
    zorder=0.8,
    extend="both"
    )

axs[1].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
for num_mod, mod in enumerate(models):
    con = axs[num_mod+2].contourf(
        vorhis_cli_ver_JJA.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-3.5e-5, 3.5e-5+3.5e-6, 3.5e-6),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod.data),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="200hPa vor reg IndR")
# %%
#   plot the climatology relative vorticity in ssp585_p3
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (6, 5))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
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
# ===================================================
con = axs[0].contourf(
    vorssp585_p3_cli_ver_JJA.mean(dim="models", skipna=True),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-3.5e-5, 3.5e-5+3.5e-6, 3.5e-6),
    zorder=0.8,
    extend="both"
    )

axs[0].format(
    rtitle="2064-2099", ltitle="MME",
)
# ===================================================
for num_mod, mod in enumerate(models):
    con = axs[num_mod+1].contourf(
        vorssp585_p3_cli_ver_JJA.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(-3.5e-5, 3.5e-5+3.5e-6, 3.5e-6),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod.data),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="200hPa vor reg IndR")
# %%
#   only plot the precipitation regress onto AIR and IndR in MME
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
# plot_array = np.reshape(range(1, 31), (6, 5))
# plot_array[5,-1] = 0
axs = fig.subplots(ncols=1, nrows=2, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
yticks = np.arange(-30, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents)

# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ===================================================
for ax in axs:
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
# ===================================================
con = axs[0].contourf(
    pre_AIR_India_pre_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
sepl.plt_sig(
    pre_AIR_India_pre_rvalue, axs[0], n, np.where(pre_AIR_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="AIR and GPCP",
)
# ===================================================
con = axs[1].contourf(
    pre_his_India_pre_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_his_India_pre_rvalue_ens, axs[1], n, np.where(pre_his_India_pre_rvalue_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   only plot the circulation regress onto AIR and IndR in MME
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
    if lev == 200.0:
        for ax in axs[num_lev, :]:
            x0 = 50
            y0 = 15
            width = 90
            height = 32.5
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
    elif lev == 850.0:
        for ax in axs[num_lev, :]:
            x0 = 110
            y0 = 15
            width = 27
            height = 22.5
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
    con = axs[num_lev, 0].contourf(
        IndRAIR_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRAIR_ERA5_hgt_slope.sel(level=lev), axs[num_lev, 0], n, np.where(IndRAIR_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_lev, 0].quiver(
        IndRAIR_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
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
        IndRAIR_ERA5_u_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        rtitle="1979-2014 {:.0f}hPa".format(lev), ltitle="AIR & ERA5",
    )
    # ======================================
    con = axs[num_lev, 1].contourf(
        IndR_his_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_ens.sel(level=lev), axs[num_lev, 1], n, np.where(IndR_his_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 1].quiver(
        IndR_his_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev)[::ski, ::ski],
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
        IndR_his_u_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
fig.format(abc="(a)", abcloc="l", suptitle="hgt&U reg IndR".format(lev))
# %%
#   plot the avalue of hgt&u&v regress onto IndR in ERA5 and historical
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 31), (6, 5))
    plot_array[-1,-1] = 0
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
    con = axs[0].contourf(
        IndRAIR_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRAIR_ERA5_hgt_slope.sel(level=lev), axs[0], n, np.where(IndRAIR_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[0].quiver(
        IndRAIR_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
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
        IndRAIR_ERA5_u_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="AIR & ERA5",
    )
    # ======================================
    con = axs[1].contourf(
        IndRGPCP_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRGPCP_ERA5_hgt_slope.sel(level=lev), axs[1], n, np.where(IndRGPCP_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndRGPCP_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
        IndRGPCP_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
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
        IndRGPCP_ERA5_u_slope.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRGPCP_ERA5_v_slope.sel(level=lev).where(IndRGPCP_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="GPCP & ERA5",
    )
    # ======================================
    con = axs[2].contourf(
        IndR_his_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_ens.sel(level=lev), axs[2], n, np.where(IndR_his_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[2].quiver(
        IndR_his_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[2].quiver(
        IndR_his_u_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[2].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+3].contourf(
            IndR_his_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_his_hgt_slope.sel(models=mod,level=lev), axs[num_mod+3], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+3].quiver(
            IndR_his_u_slope.sel(models=mod,level=lev)[::ski, ::ski],
            IndR_his_v_slope.sel(models=mod,level=lev)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="grey6",
        )

        m = axs[num_mod+3].quiver(
            IndR_his_u_slope.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            IndR_his_v_slope.sel(models=mod,level=lev).where(IndR_his_wind_mask.sel(models=mod,level=lev) > 0.0)[::ski, ::ski],
            zorder=1.1,
            headwidth=2.6,
            headlength=2.3,
            headaxislength=2.3,
            scale_units="xy",
            scale=scalelevel[num_lev],
            pivot="mid",
            color="black",
        )

        qk = axs[num_mod+3].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+3].format(
            rtitle="1979-2014", ltitle="{}".format(mod.data),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the correlation scatter-plot, x:pcc, y:corr(IndR, NCR)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
m = axs[0].scatter(1.0, IndR_GPCP_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="GPCP", marker="s", color="blue5")
for num_models, mod in enumerate(models_array):
    m = axs[0].scatter((np.array(IndR_200hgt_pcc)+np.array(IndR_850hgt_pcc))[num_models]/2.0, IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod)
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter((np.array(IndR_200hgt_pcc)+np.array(IndR_850hgt_pcc))[26]/2.0, ca.cal_rMME(IndR_his_NC_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^")
m = axs[0].scatter((np.array(IndR_200hgt_pcc)+np.array(IndR_850hgt_pcc))[27]/2.0, ca.cal_rMME(IndR_his_NC_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*")

#   x-axis title
axs[0].text(-0.90,0.03,s='pcc_mean')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, NCR)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.2), ylim=(-0.6,0.6), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
