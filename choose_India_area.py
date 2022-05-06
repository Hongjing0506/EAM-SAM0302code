'''
Author: ChenHJ
Date: 2022-05-06 15:24:33
LastEditors: ChenHJ
LastEditTime: 2022-05-06 15:46:29
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

hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True)
hgtERA5_ver_JJA = hgtERA5_ver_JJA-hgtERA5_ver_JJA.mean(dim="lon", skipna=True)
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True)
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True)
qERA5_ver_JJA = ca.p_time(qERA5, 6, 9, True)
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)

hgtERA5_ver_JJA = ca.detrend_dim(hgtERA5_ver_JJA, "time", deg=1, demean=False)
uERA5_ver_JJA = ca.detrend_dim(uERA5_ver_JJA, "time", deg=1, demean=False)
vERA5_ver_JJA = ca.detrend_dim(vERA5_ver_JJA, "time", deg=1, demean=False)
qERA5_ver_JJA = ca.detrend_dim(qERA5_ver_JJA, "time", deg=1, demean=False)
spERA5_JJA = ca.detrend_dim(spERA5_JJA, "time", deg=1, demean=False)

fhgthis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/zg_historical_r144x72_195001-201412.nc")
hgthis_ver_JJA = fhgthis_ver_JJA["zg"]
hgthis_ver_JJA = hgthis_ver_JJA - hgthis_ver_JJA.mean(dim="lon", skipna=True)

fuhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/ua_historical_r144x72_195001-201412.nc")
uhis_ver_JJA = fuhis_ver_JJA["ua"]

fvhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/va_historical_r144x72_195001-201412.nc")
vhis_ver_JJA = fvhis_ver_JJA["va"]

fhgtssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/zg_ssp585_r144x72_201501-209912.nc")
hgtssp585_ver_JJA = fhgtssp585_ver_JJA["zg"]
hgtssp585_ver_JJA = hgtssp585_ver_JJA - hgtssp585_ver_JJA.mean(dim="lon", skipna=True)

fussp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ua_ssp585_r144x72_201501-209912.nc")
ussp585_ver_JJA = fussp585_ver_JJA["ua"]

fvssp585_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/va_ssp585_r144x72_201501-209912.nc")
vssp585_ver_JJA = fvssp585_ver_JJA["va"]

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
#   calculate the precipitation in India
lat = preCRU_JJA.coords["lat"]
lon = preCRU_JJA.coords["lon"]

India_N = 32.5
India_S = 8.0
India_W = 70.0
India_E = 100.0
lat_India_range = lat[(lat >= India_S) & (lat <= India_N)]
lon_India_range = lon[(lon >= India_W) & (lon <= India_E)]

preCRU_India_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
preGPCP_India_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
prehis_India_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)
pressp585_India_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Northern China
lat_NC_range = lat[(lat>=27.5) & (lat<=37.5)]
lon_NC_range = lon[(lon>=105.0) & (lon<=125.0)]
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

#   calculate the 200hPa u-wind over the East Asia
lat_EA_range = lat[(lat>=27.5) & (lat<=37.5)]
lon_EA_range = lon[(lon>=100.0) & (lon<=125.0)]

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
preCRU_India_JJA = ca.detrend_dim(preCRU_India_JJA, "time", deg=1, demean=False)
preGPCP_India_JJA = ca.detrend_dim(preGPCP_India_JJA, "time", deg=1, demean=False)
prehis_India_JJA = ca.detrend_dim(prehis_India_JJA, "time", deg=1, demean=False)
pressp585_India_JJA = ca.detrend_dim(pressp585_India_JJA, "time", deg=1, demean=False)

preCRU_NC_JJA = ca.detrend_dim(preCRU_NC_JJA, "time", deg=1, demean=False)
preGPCP_NC_JJA = ca.detrend_dim(preGPCP_NC_JJA, "time", deg=1, demean=False)
prehis_NC_JJA = ca.detrend_dim(prehis_NC_JJA, "time", deg=1, demean=False)
pressp585_NC_JJA = ca.detrend_dim(pressp585_NC_JJA, "time", deg=1, demean=False)

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

(
    pre_his_India_pre_slope,
    pre_his_India_pre_intercept,
    pre_his_India_pre_rvalue,
    pre_his_India_pre_pvalue,
    pre_his_India_pre_hypothesis,
) = ca.dim_linregress(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979), prehis_JJA.sel(time=prehis_JJA.time.dt.year>=1979))
# %%
pre_his_India_pre_slope_ens = pre_his_India_pre_slope.mean(dim="models", skipna=True)

pre_his_India_pre_slope_ens_mask = ca.MME_reg_mask(pre_his_India_pre_slope_ens, pre_his_India_pre_slope.std(dim="models", skipna=True), len(pre_his_India_pre_slope.coords["models"]), True)

pre_his_India_pre_rvalue_ens = ca.cal_rMME(pre_his_India_pre_rvalue,"models")

pre_his_India_pre_rvalue_ens_mask = ca.MME_reg_mask(pre_his_India_pre_rvalue_ens, pre_his_India_pre_rvalue.std(dim="models", skipna=True), len(pre_his_India_pre_rvalue.coords["models"]), True)
# %%
#   plot the regression coefficients in CRU, GPCP, historical run
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
    # Inida area
    x0 = India_W
    y0 = India_S
    width = India_E - India_W
    height = India_N - India_S
    patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_CRU_India_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_CRU_India_pre_slope, axs[0], n, np.where(pre_CRU_India_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
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
    extend="both",
    )
sepl.plt_sig(
    pre_GPCP_India_pre_slope, axs[1], n, np.where(pre_GPCP_India_pre_pvalue[::n, ::n] <= 0.05), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="GPCP",
)
# ===================================================
con = axs[2].contourf(
    pre_his_India_pre_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_his_India_pre_slope_ens, axs[2], n, np.where(pre_his_India_pre_slope_ens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[2].format(
    rtitle="1979-2014", ltitle="MME",
)
# ===================================================
for num_models,mod in enumerate(pre_his_India_pre_slope.coords["models"].data):
    con = axs[num_models+3].contourf(
    pre_his_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both",
    )
    sepl.plt_sig(
        pre_his_India_pre_slope.sel(models=mod), axs[num_models+3], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 4.0,
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