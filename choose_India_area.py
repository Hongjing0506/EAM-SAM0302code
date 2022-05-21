'''
Author: ChenHJ
Date: 2022-05-06 15:24:33
LastEditors: ChenHJ
LastEditTime: 2022-05-21 22:48:46
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

fsstHad = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/HadISST_r144x72_1870-2020.nc")
sstHad = fsstHad["sst"].sel(time=(fsstHad["time"].dt.year>=1979) & (fsstHad["time"].dt.year<=2014))

hgtERA5_ver_JJA = ca.p_time(hgtERA5, 6, 8, True)
hgtERA5_ver_JJA = hgtERA5_ver_JJA-hgtERA5_ver_JJA.mean(dim="lon", skipna=True)
uERA5_ver_JJA = ca.p_time(uERA5, 6, 8, True)
vERA5_ver_JJA = ca.p_time(vERA5, 6, 8, True)
qERA5_ver_JJA = ca.p_time(qERA5, 6, 9, True)
spERA5_JJA = ca.p_time(spERA5, 6, 8, True)
wERA5_JJA = ca.p_time(wERA5, 6, 8, True)
sstHad_JJA = ca.p_time(sstHad, 6, 8, True)

hgtERA5_ver_JJA = ca.detrend_dim(hgtERA5_ver_JJA, "time", deg=1, demean=False)
uERA5_ver_JJA = ca.detrend_dim(uERA5_ver_JJA, "time", deg=1, demean=False)
vERA5_ver_JJA = ca.detrend_dim(vERA5_ver_JJA, "time", deg=1, demean=False)
qERA5_ver_JJA = ca.detrend_dim(qERA5_ver_JJA, "time", deg=1, demean=False)
spERA5_JJA = ca.detrend_dim(spERA5_JJA, "time", deg=1, demean=False)
wERA5_JJA = ca.detrend_dim(wERA5_JJA, "time", deg=1, demean=False)
sstHad_JJA = ca.detrend_dim(sstHad_JJA, "time", deg=1, demean=False)

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

#   read the SST data in observation and CMIP6
fssthis_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/tos_historical_r144x72_195001-201412.nc")
ssthis_JJA = fssthis_JJA["sst"].sel(time=fssthis_JJA["time"].dt.year>=1979)
ssthis_JJA = ca.detrend_dim(ssthis_JJA, "time", deg=1, demean=False)


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

fsstssp585_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/tos_ssp585_r144x72_201501-209912.nc")
sstssp585_JJA = fsstssp585_JJA["sst"]
sstssp585_p3_JJA = sstssp585_JJA.sel(time=sstssp585_JJA.time.dt.year>=2064)
sstssp585_JJA = ca.detrend_dim(sstssp585_JJA, "time", deg=1, demean=False)
sstssp585_p3_JJA = ca.detrend_dim(sstssp585_p3_JJA, "time", deg=1, demean=False)

#   read the temperature data in ERA5/historical/ssp585
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

ERA5_EAM = ca.EAM(uERA5_ver_JJA)
ERA5_EAM = ca.detrend_dim(ERA5_EAM, "time", deg=1, demean=False)

ERA5_IWF = ca.IWF(uERA5_ver_JJA, vERA5_ver_JJA)
ERA5_IWF = ca.detrend_dim(ERA5_IWF, "time", deg=1, demean=False)

ERA5_LKY = ca.LKY(uERA5_ver_JJA, vERA5_ver_JJA)
ERA5_LKY = ca.detrend_dim(ERA5_LKY, "time", deg=1, demean=False)

fhis_EAM = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_EAM_index_1950-2014.nc")
his_EAM = fhis_EAM["EAM"].sel(time=fhis_EAM["time"].dt.year>=1979)
his_EAM = ca.detrend_dim(his_EAM, "time", deg=1, demean=False)

fssp585_EAM = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_EAM_index_2015-2099.nc")
ssp585_EAM = fssp585_EAM["EAM"]
ssp585_p3_EAM = fssp585_EAM["EAM"].sel(time=fssp585_EAM["time"].dt.year>=2064)
ssp585_EAM = ca.detrend_dim(ssp585_EAM, "time", deg=1, demean=False)
ssp585_p3_EAM = ca.detrend_dim(ssp585_p3_EAM, "time", deg=1, demean=False)

fhis_IWF = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/his_IWF_index_1950-2014.nc")
his_IWF = fhis_IWF["IWF"].sel(time=fhis_IWF["time"].dt.year>=1979)
his_IWF = ca.detrend_dim(his_IWF, "time", deg=1, demean=False)

fssp585_IWF = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/ssp585_IWF_index_2015-2099.nc")
ssp585_IWF = fssp585_IWF["IWF"]
ssp585_p3_IWF = fssp585_IWF["IWF"].sel(time=fssp585_IWF["time"].dt.year>=2064)
ssp585_IWF = ca.detrend_dim(ssp585_IWF, "time", deg=1, demean=False)
ssp585_p3_IWF = ca.detrend_dim(ssp585_p3_IWF, "time", deg=1, demean=False)

#   read the pcc data from the files
fNCR_200hgt_pcc = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCR_200hgt_pcc.nc")
NCR_200hgt_pcc = fNCR_200hgt_pcc["pcc"]

fNCR_850hgt_pcc = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/NCR_850hgt_pcc.nc")
NCR_850hgt_pcc = fNCR_850hgt_pcc["pcc"]
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

his_LKY = ca.LKY(uhis_ver_JJA, vhis_ver_JJA)
ssp585_p3_LKY = ca.LKY(ussp585_p3_ver_JJA, vssp585_p3_ver_JJA)

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

#   calculate the north India peninsula precipitation
nIndia_N = 32.5
# nIndia_N = 30.0
nIndia_S = 17.5
nIndia_W = 70.0
nIndia_E = 86.0
lat_nIndia_range = lat[(lat >= nIndia_S) & (lat <= nIndia_N)]
lon_nIndia_range = lon[(lon >= nIndia_W) & (lon <= nIndia_E)]

preGPCP_nIndia_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range)).mean(dim="lon", skipna=True)
prehis_nIndia_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range)).mean(dim="lon", skipna=True)
pressp585_nIndia_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range)).mean(dim="lon", skipna=True)
pressp585_p3_nIndia_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range)).mean(dim="lon", skipna=True)

#   calculate the west India peninsula precipitation
wIndia_N = 32.5
# wIndia_N = 30.0
wIndia_S = 8.0
wIndia_W = 70.0
wIndia_E = 77.0
lat_wIndia_range = lat[(lat >= wIndia_S) & (lat <= wIndia_N)]
lon_wIndia_range = lon[(lon >= wIndia_W) & (lon <= wIndia_E)]

prehis_wIndia_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_wIndia_range, lon=lon_wIndia_range)).mean(dim="lon", skipna=True)
pressp585_wIndia_JJA = ca.cal_lat_weighted_mean(pressp585_JJA.sel(lat=lat_wIndia_range, lon=lon_wIndia_range)).mean(dim="lon", skipna=True)
pressp585_p3_wIndia_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_wIndia_range, lon=lon_wIndia_range)).mean(dim="lon", skipna=True)
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
EAhigh_N = 50.0
EAhigh_S = 27.5
EAhigh_W = 105.0
EAhigh_E = 140.0
lat_EAhigh_range = lat[(lat>=EAhigh_S) & (lat<=EAhigh_N)]
lon_EAhigh_range = lon[(lon>=EAhigh_W) & (lon<=EAhigh_E)]
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
WAhigh_N = 50.0
WAhigh_S = 25.0
WAhigh_W = 50.0
WAhigh_E = 80.0
lat_WAhigh_range = lat[(lat>=WAhigh_S) & (lat<=WAhigh_N)]
lon_WAhigh_range = lon[(lon>=WAhigh_W) & (lon<=WAhigh_E)]
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

#   calculate the vorticity over the West Asia in 200hPa
WNPhigh_N = 37.5
WNPhigh_S = 15.0
WNPhigh_W = 110.0
WNPhigh_E = 140.0
lat_WNPhigh_range = lat[(lat>=WNPhigh_S) & (lat<=WNPhigh_N)]
lon_WNPhigh_range = lon[(lon>=WNPhigh_W) & (lon<=WNPhigh_E)]
uERA5_WNPhigh_JJA = uERA5_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
uhis_WNPhigh_JJA = uhis_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
ussp585_WNPhigh_JJA = ussp585_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
ussp585_p3_WNPhigh_JJA = ussp585_p3_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)

vERA5_WNPhigh_JJA = vERA5_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
vhis_WNPhigh_JJA = vhis_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
vssp585_WNPhigh_JJA = vssp585_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
vssp585_p3_WNPhigh_JJA = vssp585_p3_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)

vorERA5_WNPhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uERA5_WNPhigh_JJA, vERA5_WNPhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorhis_WNPhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uhis_WNPhigh_JJA, vhis_WNPhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_WNPhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_WNPhigh_JJA, vssp585_WNPhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_p3_WNPhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_p3_WNPhigh_JJA, vssp585_p3_WNPhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
# %%
#   calculate the detrend-data for the picked-up area data
vorERA5_ver_JJA = ca.detrend_dim(vorERA5_ver_JJA, "time", deg=1, demean=False)
vorhis_ver_JJA = ca.detrend_dim(vorhis_ver_JJA, "time", deg=1, demean=False)
vorssp585_ver_JJA = ca.detrend_dim(vorssp585_ver_JJA, "time", deg=1, demean=False)
vorssp585_p3_ver_JJA = ca.detrend_dim(vorssp585_p3_ver_JJA, "time", deg=1, demean=False)

his_LKY = ca.detrend_dim(his_LKY, "time", deg=1, demean=False)
ssp585_p3_LKY = ca.detrend_dim(ssp585_p3_LKY, "time", deg=1, demean=False)

# preCRU_India_JJA = ca.detrend_dim(preCRU_India_JJA, "time", deg=1, demean=False)
preGPCP_India_JJA = ca.detrend_dim(preGPCP_India_JJA, "time", deg=1, demean=False)
prehis_India_JJA = ca.detrend_dim(prehis_India_JJA, "time", deg=1, demean=False)
pressp585_India_JJA = ca.detrend_dim(pressp585_India_JJA, "time", deg=1, demean=False)
pressp585_p3_India_JJA = ca.detrend_dim(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064), "time", deg=1, demean=False)

prehis_nIndia_JJA = ca.detrend_dim(prehis_nIndia_JJA, "time", deg=1, demean=False)
pressp585_nIndia_JJA = ca.detrend_dim(pressp585_nIndia_JJA, "time", deg=1, demean=False)
pressp585_p3_nIndia_JJA = ca.detrend_dim(pressp585_nIndia_JJA.sel(time=pressp585_nIndia_JJA.time.dt.year>=2064), "time", deg=1, demean=False)

prehis_wIndia_JJA = ca.detrend_dim(prehis_wIndia_JJA, "time", deg=1, demean=False)
pressp585_wIndia_JJA = ca.detrend_dim(pressp585_wIndia_JJA, "time", deg=1, demean=False)
pressp585_p3_wIndia_JJA = ca.detrend_dim(pressp585_wIndia_JJA.sel(time=pressp585_wIndia_JJA.time.dt.year>=2064), "time", deg=1, demean=False)

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

vorERA5_WNPhigh_JJA = ca.detrend_dim(vorERA5_WNPhigh_JJA, "time", deg=1, demean=False)
vorhis_WNPhigh_JJA = ca.detrend_dim(vorhis_WNPhigh_JJA, "time", deg=1, demean=False)
vorssp585_WNPhigh_JJA = ca.detrend_dim(vorssp585_WNPhigh_JJA, "time", deg=1, demean=False)
vorssp585_p3_WNPhigh_JJA = ca.detrend_dim(vorssp585_p3_WNPhigh_JJA, "time", deg=1, demean=False)
# %%
#   calculate the precipitation fields regression onto IndR
(
    pre_GPCP_India_pre_slope,
    pre_GPCP_India_pre_intercept,
    pre_GPCP_India_pre_rvalue,
    pre_GPCP_India_pre_pvalue,
    pre_GPCP_India_pre_hypothesis,
) = ca.dim_linregress(preGPCP_India_JJA, preGPCP_JJA)

preAIR_JJA.coords["time"] = preGPCP_JJA.coords["time"]
(
    pre_AIR_India_pre_slope,
    pre_AIR_India_pre_intercept,
    pre_AIR_India_pre_rvalue,
    pre_AIR_India_pre_pvalue,
    pre_AIR_India_pre_hypothesis,
) = ca.dim_linregress(preAIR_JJA, preGPCP_JJA)

(
    pre_his_India_pre_slope,
    pre_his_India_pre_intercept,
    pre_his_India_pre_rvalue,
    pre_his_India_pre_pvalue,
    pre_his_India_pre_hypothesis,
) = ca.dim_linregress(prehis_India_JJA, prehis_JJA)

(
    pre_ssp585_p3_India_pre_slope,
    pre_ssp585_p3_India_pre_intercept,
    pre_ssp585_p3_India_pre_rvalue,
    pre_ssp585_p3_India_pre_pvalue,
    pre_ssp585_p3_India_pre_hypothesis,
) = ca.dim_linregress(pressp585_p3_India_JJA, pressp585_p3_JJA)
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
# #   calculate the hgt/u/v regression onto IndR in ERA5, historical, ssp585, ssp585_p3
# preGPCP_India_JJA.coords["time"] = hgtERA5_ver_JJA.sel(time=hgtERA5_ver_JJA.time.dt.year>=1979).coords["time"]
# preAIR_JJA.coords["time"] = hgtERA5_ver_JJA.coords["time"]


# (
#     IndR_his_hgt_slope,
#     IndR_his_hgt_intercept,
#     IndR_his_hgt_rvalue,
#     IndR_his_hgt_pvalue,
#     IndR_his_hgt_hypothesis,
# ) = ca.dim_linregress(prehis_India_JJA, hgthis_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndR_his_u_slope,
#     IndR_his_u_intercept,
#     IndR_his_u_rvalue,
#     IndR_his_u_pvalue,
#     IndR_his_u_hypothesis,
# ) = ca.dim_linregress(prehis_India_JJA, uhis_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndR_his_v_slope,
#     IndR_his_v_intercept,
#     IndR_his_v_rvalue,
#     IndR_his_v_pvalue,
#     IndR_his_v_hypothesis,
# ) = ca.dim_linregress(prehis_India_JJA, vhis_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndR_ssp585_p3_hgt_slope,
#     IndR_ssp585_p3_hgt_intercept,
#     IndR_ssp585_p3_hgt_rvalue,
#     IndR_ssp585_p3_hgt_pvalue,
#     IndR_ssp585_p3_hgt_hypothesis,
# ) = ca.dim_linregress(pressp585_p3_India_JJA, hgtssp585_p3_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndR_ssp585_p3_u_slope,
#     IndR_ssp585_p3_u_intercept,
#     IndR_ssp585_p3_u_rvalue,
#     IndR_ssp585_p3_u_pvalue,
#     IndR_ssp585_p3_u_hypothesis,
# ) = ca.dim_linregress(pressp585_p3_India_JJA, ussp585_p3_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndR_ssp585_p3_v_slope,
#     IndR_ssp585_p3_v_intercept,
#     IndR_ssp585_p3_v_rvalue,
#     IndR_ssp585_p3_v_pvalue,
#     IndR_ssp585_p3_v_hypothesis,
# ) = ca.dim_linregress(pressp585_p3_India_JJA, vssp585_p3_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndRAIR_ERA5_hgt_slope,
#     IndRAIR_ERA5_hgt_intercept,
#     IndRAIR_ERA5_hgt_rvalue,
#     IndRAIR_ERA5_hgt_pvalue,
#     IndRAIR_ERA5_hgt_hypothesis,
# ) = ca.dim_linregress(preAIR_JJA, hgtERA5_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndRAIR_ERA5_u_slope,
#     IndRAIR_ERA5_u_intercept,
#     IndRAIR_ERA5_u_rvalue,
#     IndRAIR_ERA5_u_pvalue,
#     IndRAIR_ERA5_u_hypothesis,
# ) = ca.dim_linregress(preAIR_JJA, uERA5_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndRAIR_ERA5_v_slope,
#     IndRAIR_ERA5_v_intercept,
#     IndRAIR_ERA5_v_rvalue,
#     IndRAIR_ERA5_v_pvalue,
#     IndRAIR_ERA5_v_hypothesis,
# ) = ca.dim_linregress(preAIR_JJA, vERA5_ver_JJA.sel(level=[200.0, 500.0, 850.0]))

# (
#     IndRAIR_ERA5_vor_slope,
#     IndRAIR_ERA5_vor_intercept,
#     IndRAIR_ERA5_vor_rvalue,
#     IndRAIR_ERA5_vor_pvalue,
#     IndRAIR_ERA5_vor_hypothesis,
# ) = ca.dim_linregress(preAIR_JJA, vorERA5_ver_JJA)

# (
#     IndR_his_vor_slope,
#     IndR_his_vor_intercept,
#     IndR_his_vor_rvalue,
#     IndR_his_vor_pvalue,
#     IndR_his_vor_hypothesis,
# ) = ca.dim_linregress(prehis_India_JJA, vorhis_ver_JJA)

# (
#     IndR_ssp585_p3_vor_slope,
#     IndR_ssp585_p3_vor_intercept,
#     IndR_ssp585_p3_vor_rvalue,
#     IndR_ssp585_p3_vor_pvalue,
#     IndR_ssp585_p3_vor_hypothesis,
# ) = ca.dim_linregress(pressp585_p3_India_JJA, vorssp585_p3_ver_JJA)
# # %%
# #   save the regression results
# level=IndR_his_hgt_slope.coords["level"]
# lat=IndR_his_hgt_slope.coords["lat"]
# lon=IndR_his_hgt_slope.coords["lon"]

# IndRAIR_ERA5_hgt_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_slope.data),
#         intercept=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_intercept.data),
#         rvalue=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_rvalue.data),
#         pvalue=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_pvalue.data),
#         hypothesis=(["level", "lat", "lon"], IndRAIR_ERA5_hgt_hypothesis.data),
#     ),
#     coords=dict(
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="hgt fields of ERA5 regress onto 1979-2014 AIR"),
# )

# IndRAIR_ERA5_u_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["level", "lat", "lon"], IndRAIR_ERA5_u_slope.data),
#         intercept=(["level", "lat", "lon"], IndRAIR_ERA5_u_intercept.data),
#         rvalue=(["level", "lat", "lon"], IndRAIR_ERA5_u_rvalue.data),
#         pvalue=(["level", "lat", "lon"], IndRAIR_ERA5_u_pvalue.data),
#         hypothesis=(["level", "lat", "lon"], IndRAIR_ERA5_u_hypothesis.data),
#     ),
#     coords=dict(
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="u fields of ERA5 regress onto 1979-2014 AIR"),
# )

# IndRAIR_ERA5_v_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["level", "lat", "lon"], IndRAIR_ERA5_v_slope.data),
#         intercept=(["level", "lat", "lon"], IndRAIR_ERA5_v_intercept.data),
#         rvalue=(["level", "lat", "lon"], IndRAIR_ERA5_v_rvalue.data),
#         pvalue=(["level", "lat", "lon"], IndRAIR_ERA5_v_pvalue.data),
#         hypothesis=(["level", "lat", "lon"], IndRAIR_ERA5_v_hypothesis.data),
#     ),
#     coords=dict(
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="v fields of ERA5 regress onto 1979-2014 AIR"),
# )

# IndR_his_hgt_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "level", "lat", "lon"], IndR_his_hgt_slope.data),
#         intercept=(["models", "level", "lat", "lon"], IndR_his_hgt_intercept.data),
#         rvalue=(["models", "level", "lat", "lon"], IndR_his_hgt_rvalue.data),
#         pvalue=(["models", "level", "lat", "lon"], IndR_his_hgt_pvalue.data),
#         hypothesis=(["models", "level", "lat", "lon"], IndR_his_hgt_hypothesis.data),
#     ),
#     coords=dict(
#         models=models.data,
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="hgt fields of historical regress onto 1979-2014 IndR"),
# )

# IndR_his_u_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "level", "lat", "lon"], IndR_his_u_slope.data),
#         intercept=(["models", "level", "lat", "lon"], IndR_his_u_intercept.data),
#         rvalue=(["models", "level", "lat", "lon"], IndR_his_u_rvalue.data),
#         pvalue=(["models", "level", "lat", "lon"], IndR_his_u_pvalue.data),
#         hypothesis=(["models", "level", "lat", "lon"], IndR_his_u_hypothesis.data),
#     ),
#     coords=dict(
#         models=models.data,
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="u fields of historical regress onto 1979-2014 IndR"),
# )

# IndR_his_v_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "level", "lat", "lon"], IndR_his_v_slope.data),
#         intercept=(["models", "level", "lat", "lon"], IndR_his_v_intercept.data),
#         rvalue=(["models", "level", "lat", "lon"], IndR_his_v_rvalue.data),
#         pvalue=(["models", "level", "lat", "lon"], IndR_his_v_pvalue.data),
#         hypothesis=(["models", "level", "lat", "lon"], IndR_his_v_hypothesis.data),
#     ),
#     coords=dict(
#         models=models.data,
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="v fields of historical regress onto 1979-2014 IndR"),
# )

# IndR_ssp585_p3_hgt_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_slope.data),
#         intercept=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_intercept.data),
#         rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_rvalue.data),
#         pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_pvalue.data),
#         hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_p3_hgt_hypothesis.data),
#     ),
#     coords=dict(
#         models=models.data,
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="hgt fields of ssp585_p3 regress onto 2064-2099 IndR"),
# )

# IndR_ssp585_p3_u_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_slope.data),
#         intercept=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_intercept.data),
#         rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_rvalue.data),
#         pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_pvalue.data),
#         hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_p3_u_hypothesis.data),
#     ),
#     coords=dict(
#         models=models.data,
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="u fields of ssp585_p3 regress onto 2064-2099 IndR"),
# )

# IndR_ssp585_p3_v_regress = xr.Dataset(
#     data_vars=dict(
#         slope=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_slope.data),
#         intercept=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_intercept.data),
#         rvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_rvalue.data),
#         pvalue=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_pvalue.data),
#         hypothesis=(["models", "level", "lat", "lon"], IndR_ssp585_p3_v_hypothesis.data),
#     ),
#     coords=dict(
#         models=models.data,
#         level=level.data,
#         lat=lat.data,
#         lon=lon.data,
#     ),
#     attrs=dict(description="v fields of ssp585_p3 regress onto 2064-2099 IndR"),
# )

# IndRAIR_ERA5_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRAIR_ERA5_hgt_regress.nc")
# IndRAIR_ERA5_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRAIR_ERA5_u_regress.nc")
# IndRAIR_ERA5_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRAIR_ERA5_v_regress.nc")

# IndR_his_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_hgt_regress.nc")
# IndR_his_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_u_regress.nc")
# IndR_his_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_v_regress.nc")

# IndR_ssp585_p3_hgt_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_hgt_regress.nc")
# IndR_ssp585_p3_u_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_u_regress.nc")
# IndR_ssp585_p3_v_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_v_regress.nc")
# %%
#   read the regression data

IndRAIR_ERA5_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRAIR_ERA5_hgt_regress.nc")
IndRAIR_ERA5_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRAIR_ERA5_u_regress.nc")
IndRAIR_ERA5_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndRAIR_ERA5_v_regress.nc")

IndR_his_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_hgt_regress.nc")
IndR_his_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_u_regress.nc")
IndR_his_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_v_regress.nc")

IndR_ssp585_p3_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_hgt_regress.nc")
IndR_ssp585_p3_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_u_regress.nc")
IndR_ssp585_p3_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_v_regress.nc")

IndRAIR_ERA5_hgt_slope = IndRAIR_ERA5_hgt_regress["slope"]
IndRAIR_ERA5_u_slope = IndRAIR_ERA5_u_regress["slope"]
IndRAIR_ERA5_v_slope = IndRAIR_ERA5_v_regress["slope"]
IndRAIR_ERA5_hgt_rvalue = IndRAIR_ERA5_hgt_regress["rvalue"]
IndRAIR_ERA5_u_rvalue = IndRAIR_ERA5_u_regress["rvalue"]
IndRAIR_ERA5_v_rvalue = IndRAIR_ERA5_v_regress["rvalue"]
IndRAIR_ERA5_hgt_pvalue = IndRAIR_ERA5_hgt_regress["pvalue"]
IndRAIR_ERA5_u_pvalue = IndRAIR_ERA5_u_regress["pvalue"]
IndRAIR_ERA5_v_pvalue = IndRAIR_ERA5_v_regress["pvalue"]

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
IndRAIR_ERA5_wind_mask = ca.wind_check(
    xr.where(IndRAIR_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRAIR_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRAIR_ERA5_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndRAIR_ERA5_v_pvalue <= 0.05, 1.0, 0.0),
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
#   calculate the sst regress onto IndR in ERA5, historical and ssp585
sstHad_JJA.coords["time"] = preAIR_JJA.coords["time"]
(
    IndR_Had_sst_slope,
    IndR_Had_sst_intercept,
    IndR_Had_sst_rvalue,
    IndR_Had_sst_pvalue,
    IndR_Had_sst_hypothesis,
) = ca.dim_linregress(preAIR_JJA, sstHad_JJA)

(
    IndR_his_sst_slope,
    IndR_his_sst_intercept,
    IndR_his_sst_rvalue,
    IndR_his_sst_pvalue,
    IndR_his_sst_hypothesis,
) = ca.dim_linregress(prehis_India_JJA, ssthis_JJA)

(
    IndR_ssp585_p3_sst_slope,
    IndR_ssp585_p3_sst_intercept,
    IndR_ssp585_p3_sst_rvalue,
    IndR_ssp585_p3_sst_pvalue,
    IndR_ssp585_p3_sst_hypothesis,
) = ca.dim_linregress(pressp585_p3_India_JJA, sstssp585_p3_JJA)

#   save the result of the sst regression
lat=IndR_Had_sst_slope.coords["lat"]
lon=IndR_Had_sst_slope.coords["lon"]

IndR_Had_sst_regress = xr.Dataset(
    data_vars=dict(
        slope=(["lat", "lon"], IndR_Had_sst_slope.data),
        intercept=(["lat", "lon"], IndR_Had_sst_intercept.data),
        rvalue=(["lat", "lon"], IndR_Had_sst_rvalue.data),
        pvalue=(["lat", "lon"], IndR_Had_sst_pvalue.data),
        hypothesis=(["lat", "lon"], IndR_Had_sst_hypothesis.data),
    ),
    coords=dict(
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="sst fields of HadISST regress onto 1979-2014 AIR"),
)

IndR_his_sst_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], IndR_his_sst_slope.data),
        intercept=(["models", "lat", "lon"], IndR_his_sst_intercept.data),
        rvalue=(["models", "lat", "lon"], IndR_his_sst_rvalue.data),
        pvalue=(["models", "lat", "lon"], IndR_his_sst_pvalue.data),
        hypothesis=(["models", "lat", "lon"], IndR_his_sst_hypothesis.data),
    ),
    coords=dict(
        models=models_array,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="sst fields of historical regress onto 1979-2014 IndR"),
)

IndR_ssp585_p3_sst_regress = xr.Dataset(
    data_vars=dict(
        slope=(["models", "lat", "lon"], IndR_ssp585_p3_sst_slope.data),
        intercept=(["models", "lat", "lon"], IndR_ssp585_p3_sst_intercept.data),
        rvalue=(["models", "lat", "lon"], IndR_ssp585_p3_sst_rvalue.data),
        pvalue=(["models", "lat", "lon"], IndR_ssp585_p3_sst_pvalue.data),
        hypothesis=(["models", "lat", "lon"], IndR_ssp585_p3_sst_hypothesis.data),
    ),
    coords=dict(
        models=models_array,
        lat=lat.data,
        lon=lon.data,
    ),
    attrs=dict(description="sst fields of ssp585_p3 regress onto 2064-2099 IndR"),
)

IndR_Had_sst_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_Had_sst_regress.nc")
IndR_his_sst_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_sst_regress.nc")
IndR_ssp585_p3_sst_regress.to_netcdf("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_sst_regress.nc")
# %%
#   read the sst regression data
IndR_Had_sst_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_Had_sst_regress.nc")
IndR_his_sst_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/detrend/nIndR_his_sst_regress.nc")
IndR_ssp585_p3_sst_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/detrend/nIndR_ssp585_p3_sst_regress.nc")

IndR_Had_sst_slope = IndR_Had_sst_regress["slope"]

IndR_Had_sst_rvalue = IndR_Had_sst_regress["rvalue"]

IndR_Had_sst_pvalue = IndR_Had_sst_regress["pvalue"]

IndR_his_sst_slope = IndR_his_sst_regress["slope"]

IndR_his_sst_rvalue = IndR_his_sst_regress["rvalue"]

IndR_his_sst_pvalue = IndR_his_sst_regress["pvalue"]

IndR_ssp585_p3_sst_slope = IndR_ssp585_p3_sst_regress["slope"]

IndR_ssp585_p3_sst_rvalue = IndR_ssp585_p3_sst_regress["rvalue"]

IndR_ssp585_p3_sst_pvalue = IndR_ssp585_p3_sst_regress["pvalue"]

IndR_his_sst_slope_ens = IndR_his_sst_slope.mean(dim="models", skipna=True)
IndR_his_sst_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_sst_slope_ens, IndR_his_sst_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_sst_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_sst_slope_ens = IndR_ssp585_p3_sst_slope.mean(dim="models", skipna=True)
IndR_ssp585_p3_sst_slope_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_sst_slope_ens, IndR_ssp585_p3_sst_slope.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_sst_slope)) >= 2.0, 1.0, 0.0)

IndR_his_sst_rvalue_ens = ca.cal_rMME(IndR_his_sst_rvalue, "models")
IndR_his_sst_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_his_sst_rvalue_ens, IndR_his_sst_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_his_sst_slope)) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_sst_rvalue_ens = ca.cal_rMME(IndR_ssp585_p3_sst_rvalue, "models")
IndR_ssp585_p3_sst_rvalue_ens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_sst_rvalue_ens, IndR_ssp585_p3_sst_rvalue.std(dim="models", skipna=True), len(models), True) + ca.cal_mmemask(IndR_ssp585_p3_sst_slope)) >= 2.0, 1.0, 0.0)

IndR_diff_sst_slope = IndR_ssp585_p3_sst_slope - IndR_his_sst_slope
IndR_diff_sst_slope_ens = IndR_diff_sst_slope.mean(dim="models", skipna=True)
IndR_diff_sst_ens_mask = ca.cal_mmemask(IndR_diff_sst_slope)

IndR_diff_sst_rvalue = ca.cal_rdiff(IndR_ssp585_p3_sst_rvalue, IndR_his_sst_rvalue)
IndR_diff_sst_rvalue_ens = ca.cal_rMME(IndR_diff_sst_rvalue, "models")

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
IndR_diff_hgt_slope_ens = IndR_diff_hgt_slope.mean(dim="models", skipna=True)
IndR_diff_hgt_slope_gens = IndR_diff_hgt_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_hgt_ens_mask = ca.cal_mmemask(IndR_diff_hgt_slope)
IndR_diff_hgt_gens_mask = ca.cal_mmemask(IndR_diff_hgt_slope.sel(models=gmodels))

IndR_diff_u_slope = IndR_ssp585_p3_u_slope - IndR_his_u_slope
IndR_diff_u_slope_ens = IndR_diff_u_slope.mean(dim="models", skipna=True)
IndR_diff_u_slope_gens = IndR_diff_u_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_u_ens_mask = ca.cal_mmemask(IndR_diff_u_slope)
IndR_diff_u_gens_mask = ca.cal_mmemask(IndR_diff_u_slope.sel(models=gmodels))

IndR_diff_v_slope = IndR_ssp585_p3_v_slope - IndR_his_v_slope
IndR_diff_v_slope_ens = IndR_diff_v_slope.mean(dim="models", skipna=True)
IndR_diff_v_slope_gens = IndR_diff_v_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_v_ens_mask = ca.cal_mmemask(IndR_diff_v_slope)
IndR_diff_v_gens_mask = ca.cal_mmemask(IndR_diff_v_slope.sel(models=gmodels))

IndR_diff_wind_ens_mask = ca.wind_check(
    xr.where(IndR_diff_u_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_u_ens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_diff_v_ens_mask > 0.0, 1.0, 0.0),
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
#   plot the circulation regress onto IndR in good-models for corr coeff.
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 9), (2, 4))
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
        IndR_his_hgt_rvalue_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_his_hgt_rvalue_gens.sel(level=lev), axs[1], n, np.where(IndR_his_hgt_rvalue_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[1].quiver(
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

    m = axs[1].quiver(
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

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+2].contourf(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+2], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+2].quiver(
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

        m = axs[num_mod+2].quiver(
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

        qk = axs[num_mod+2].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+2].format(
            rtitle="1979-2014", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the circulation regress onto IndR in good-models for reg coeff.
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 9), (2, 4))
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
        IndR_his_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_gens.sel(level=lev), axs[1], n, np.where(IndR_his_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[1].quiver(
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

    m = axs[1].quiver(
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

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="gMME",
    )
    # ======================================
    for num_mod, mod in enumerate(gmodels):
        con = axs[num_mod+2].contourf(
            IndR_his_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_his_hgt_slope.sel(models=mod,level=lev), axs[num_mod+2], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+2].quiver(
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

        m = axs[num_mod+2].quiver(
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

        qk = axs[num_mod+2].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+2].format(
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
plot_array = np.reshape(range(1, 9), (2, 4))
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
    pre_his_India_pre_rvalue_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_his_India_pre_rvalue_gens, axs[1], n, np.where(pre_his_India_pre_rvalue_gens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="gMME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+2].contourf(
    pre_his_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_his_India_pre_rvalue.sel(models=mod), axs[num_models+2], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.10), "bright purple", 4.0,
    )

    axs[num_models+2].format(
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
plot_array = np.reshape(range(1, 9), (2, 4))
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
    pre_AIR_India_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_AIR_India_pre_slope, axs[0], n, np.where(pre_AIR_India_pre_pvalue[::n, ::n] <= 0.10), "bright purple", 4.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="AIR",
)

# ===================================================
con = axs[1].contourf(
    pre_his_India_pre_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_his_India_pre_slope_gens, axs[1], n, np.where(pre_his_India_pre_slope_gens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="gMME",
)
# ===================================================
for num_models,mod in enumerate(gmodels):
    con = axs[num_models+2].contourf(
    pre_his_India_pre_slope.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-2.0, 2.1, 0.1),
    zorder=0.8,
    extend="both"
    )
    sepl.plt_sig(
        pre_his_India_pre_slope.sel(models=mod), axs[num_models+2], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.10), "bright purple", 4.0,
    )

    axs[num_models+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")
# %%
#   plot the circulation regress on IndR in good-models for corr coeff. in ssp585_p3
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
#   plot the circulation regress on IndR in good-models for reg coeff. in ssp585_p3
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
#   plot the circulation regress onto IndR in good-models for corr coeff. in diff
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
#   plot the circulation regress onto IndR in good-models for reg coeff. in diff
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
IndR_GPCP_NC_regress = stats.linregress(preAIR_JJA, preGPCP_NC_JJA)

IndR_his_NC_regress = ca.dim_linregress(prehis_India_JJA, prehis_NC_JJA)

IndR_ssp585_p3_NC_regress = ca.dim_linregress(pressp585_p3_India_JJA, pressp585_p3_NC_JJA)

IndR_diff_NC_slope = IndR_ssp585_p3_NC_regress[0] - IndR_his_NC_regress[0]
IndR_diff_NC_rvalue = ca.cal_rdiff(IndR_ssp585_p3_NC_regress[2], IndR_his_NC_regress[2])

#   KP
IndR_GPCP_KP_regress = stats.linregress(preAIR_JJA, preGPCP_KP_JJA)

IndR_his_KP_regress = ca.dim_linregress(prehis_India_JJA, prehis_KP_JJA)

IndR_ssp585_p3_KP_regress = ca.dim_linregress(pressp585_p3_India_JJA, pressp585_p3_KP_JJA)

IndR_diff_KP_slope = IndR_ssp585_p3_KP_regress[0] - IndR_his_KP_regress[0]
IndR_diff_KP_rvalue = ca.cal_rdiff(IndR_ssp585_p3_KP_regress[2], IndR_his_KP_regress[2])

#   SJ (actually Southern Japan and Korean peninsula)
IndR_GPCP_SJ_regress = stats.linregress(preAIR_JJA, preGPCP_SJ_JJA)

IndR_his_SJ_regress = ca.dim_linregress(prehis_India_JJA, prehis_SJ_JJA)

IndR_ssp585_p3_SJ_regress = ca.dim_linregress(pressp585_p3_India_JJA, pressp585_p3_SJ_JJA)

IndR_diff_SJ_slope = IndR_ssp585_p3_SJ_regress[0] - IndR_his_SJ_regress[0]
IndR_diff_SJ_rvalue = ca.cal_rdiff(IndR_ssp585_p3_SJ_regress[2], IndR_his_SJ_regress[2])

#   EAhigh
IndR_GPCP_EAhigh_regress = stats.linregress(preAIR_JJA, vorERA5_EAhigh_JJA)
IndR_his_EAhigh_regress = ca.dim_linregress(prehis_India_JJA, vorhis_EAhigh_JJA)
IndR_ssp585_p3_EAhigh_regress = ca.dim_linregress(pressp585_p3_India_JJA, vorssp585_p3_EAhigh_JJA)

IndR_diff_EAhigh_slope = IndR_ssp585_p3_EAhigh_regress[0] - IndR_his_EAhigh_regress[0]
IndR_diff_EAhigh_rvalue = ca.cal_rdiff(IndR_ssp585_p3_EAhigh_regress[2], IndR_his_EAhigh_regress[2])

#   WNPhigh
IndR_GPCP_WNPhigh_regress = stats.linregress(preAIR_JJA, vorERA5_WNPhigh_JJA)
IndR_his_WNPhigh_regress = ca.dim_linregress(prehis_India_JJA, vorhis_WNPhigh_JJA)
IndR_ssp585_p3_WNPhigh_regress = ca.dim_linregress(pressp585_p3_India_JJA, vorssp585_p3_WNPhigh_JJA)

IndR_diff_WNPhigh_slope = IndR_ssp585_p3_WNPhigh_regress[0] - IndR_his_WNPhigh_regress[0]
IndR_diff_WNPhigh_rvalue = ca.cal_rdiff(IndR_ssp585_p3_WNPhigh_regress[2], IndR_his_WNPhigh_regress[2])

#   EAhigh
IndR_GPCP_WAhigh_regress = stats.linregress(preAIR_JJA, vorERA5_WAhigh_JJA)
IndR_his_WAhigh_regress = ca.dim_linregress(prehis_India_JJA, vorhis_WAhigh_JJA)
IndR_ssp585_p3_WAhigh_regress = ca.dim_linregress(pressp585_p3_India_JJA, vorssp585_p3_WAhigh_JJA)

IndR_diff_WAhigh_slope = IndR_ssp585_p3_WAhigh_regress[0] - IndR_his_WAhigh_regress[0]
IndR_diff_WAhigh_rvalue = ca.cal_rdiff(IndR_ssp585_p3_WAhigh_regress[2], IndR_his_WAhigh_regress[2])

#   EAM index
IndR_GPCP_EAM_regress = stats.linregress(preAIR_JJA, ERA5_EAM)
IndR_his_EAM_regress = ca.dim_linregress(prehis_India_JJA, his_EAM)
IndR_ssp585_p3_EAM_regress = ca.dim_linregress(pressp585_p3_India_JJA, ssp585_p3_EAM)
IndR_diff_EAM_slope = IndR_ssp585_p3_EAM_regress[0] - IndR_his_EAM_regress[0]
IndR_diff_EAM_rvalue = ca.cal_rdiff(IndR_ssp585_p3_EAM_regress[2], IndR_his_EAM_regress[2])

#   IWF index
IndR_GPCP_IWF_regress = stats.linregress(preAIR_JJA, ERA5_IWF)
IndR_his_IWF_regress = ca.dim_linregress(prehis_India_JJA, his_IWF)
IndR_ssp585_p3_IWF_regress = ca.dim_linregress(pressp585_p3_India_JJA, ssp585_p3_IWF)
IndR_diff_IWF_slope = IndR_ssp585_p3_IWF_regress[0] - IndR_his_IWF_regress[0]
IndR_diff_IWF_rvalue = ca.cal_rdiff(IndR_ssp585_p3_IWF_regress[2], IndR_his_IWF_regress[2])

#   LKY index
IndR_GPCP_LKY_regress = stats.linregress(preAIR_JJA, ERA5_LKY)
IndR_his_LKY_regress = ca.dim_linregress(prehis_India_JJA, his_LKY)
IndR_ssp585_p3_LKY_regress = ca.dim_linregress(pressp585_p3_India_JJA, ssp585_p3_LKY)
IndR_diff_LKY_slope = IndR_ssp585_p3_LKY_regress[0] - IndR_his_LKY_regress[0]
IndR_diff_LKY_rvalue = ca.cal_rdiff(IndR_ssp585_p3_LKY_regress[2], IndR_his_LKY_regress[2])
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
fig.format(suptitle="corr(IndR, NCR)")
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
fig.format(suptitle="Reg. Coeff. IndR and NCR")
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
fig.format(suptitle="Corr. Coeff. IndR and NCR")
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
        IndR_his_hgt_rvalue_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_his_hgt_rvalue_ens.sel(level=lev), axs[1], n, np.where(IndR_his_hgt_rvalue_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[1].quiver(
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

    m = axs[1].quiver(
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

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+2].contourf(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
        )
        sepl.plt_sig(
            IndR_his_hgt_rvalue.sel(models=mod,level=lev), axs[num_mod+2], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+2].quiver(
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

        m = axs[num_mod+2].quiver(
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

        qk = axs[num_mod+2].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+2].format(
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
plot_array[5,-2:] = 0
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
for num_models,mod in enumerate(pre_his_India_pre_slope.coords["models"].data):
    con = axs[num_models+2].contourf(
    pre_his_India_pre_rvalue.sel(models=mod),
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    )
    sepl.plt_sig(
        pre_his_India_pre_rvalue.sel(models=mod), axs[num_models+2], n, np.where(pre_his_India_pre_pvalue.sel(models=mod)[::n, ::n] <= 0.10), "bright purple", 4.0,
    )

    axs[num_models+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="pre reg IndR")


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
#   plot the correlation scatter-plot, x:pcc, y:corr(IndR, KP)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
m = axs[0].scatter(1.0, IndR_GPCP_KP_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="GPCP", marker="s", color="blue5")
for num_models, mod in enumerate(models_array):
    m = axs[0].scatter((np.array(IndR_200hgt_pcc)+np.array(IndR_850hgt_pcc))[num_models]/2.0, IndR_his_KP_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod)
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter((np.array(IndR_200hgt_pcc)+np.array(IndR_850hgt_pcc))[26]/2.0, ca.cal_rMME(IndR_his_KP_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^")
m = axs[0].scatter((np.array(IndR_200hgt_pcc)+np.array(IndR_850hgt_pcc))[27]/2.0, ca.cal_rMME(IndR_his_KP_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*")
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
            #   EAM index
            x0 = 110.0
            y0 = 40.0
            width = 40.0
            height = 10.0
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="green8", linestyle="-")
            x0 = 110.0
            y0 = 25.0
            width = 40.0
            height = 10.0
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="green8", linestyle="-")
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
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 31), (6, 5))
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
        IndR_his_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_ens.sel(level=lev), axs[1], n, np.where(IndR_his_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[1].quiver(
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

    m = axs[1].quiver(
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

    qk = axs[1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[1].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    for num_mod, mod in enumerate(models):
        con = axs[num_mod+2].contourf(
            IndR_his_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_his_hgt_slope.sel(models=mod,level=lev), axs[num_mod+2], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+2].quiver(
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

        m = axs[num_mod+2].quiver(
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

        qk = axs[num_mod+2].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+2].format(
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
#   only plot the precipitation regress onto AIR and IndR in gMME
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
    pre_his_India_pre_rvalue_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.0, 1.1, 0.1),
    zorder=0.8,
    
    )
sepl.plt_sig(
    pre_his_India_pre_rvalue_gens, axs[1], n, np.where(pre_his_India_pre_rvalue_gens_mask[::n, ::n] > 0.0), "bright purple", 4.0,
)

axs[1].format(
    rtitle="1979-2014", ltitle="gMME",
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
plot_array = np.reshape(range(1, 13), (3, 4))
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
        IndR_his_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_gens.sel(level=lev), axs[num_lev, 1], n, np.where(IndR_his_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 1].quiver(
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

    m = axs[num_lev, 1].quiver(
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

    qk = axs[num_lev, 1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 1].format(
        rtitle="1979-2014 {:.0f}hPa".format(lev), ltitle="gMME",
    )
# ======================================
    con = axs[num_lev, 2].contourf(
        IndR_ssp585_p3_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_slope_gens.sel(level=lev), axs[num_lev, 2], n, np.where(IndR_ssp585_p3_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 2].quiver(
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

    m = axs[num_lev, 2].quiver(
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

    qk = axs[num_lev, 2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 2].format(
        rtitle="2064-2099 {:.0f}hPa".format(lev), ltitle="gMME",
    )
# ======================================
    con = axs[num_lev, 3].contourf(
        IndR_diff_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_diff_hgt_slope_gens.sel(level=lev), axs[num_lev, 3], n, np.where(IndR_diff_hgt_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 3].quiver(
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

    m = axs[num_lev, 3].quiver(
        IndR_diff_u_slope_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_diff_v_slope_gens.sel(level=lev).where(IndR_diff_wind_gens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 3].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 3].format(
        rtitle="diff {:.0f}hPa".format(lev), ltitle="gMME",
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="hgt&U reg IndR".format(lev))
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
#   plot the correlation scatter-plot, x:corr(IndR, EAhigh), y:corr(IndR, NCR)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']
m = axs[0].scatter(IndR_GPCP_EAhigh_regress[2], IndR_GPCP_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="GPCP", marker="s", color="blue5", ec="black")

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_his_EAhigh_regress[2].sel(models=mod), IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, ec="black")
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_his_EAhigh_regress[2].sel(models=mod), IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, marker="h", ec="black")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(ca.cal_rMME(IndR_his_EAhigh_regress[2], "models"), ca.cal_rMME(IndR_his_NC_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^", ec="black")
m = axs[0].scatter(ca.cal_rMME(IndR_his_EAhigh_regress[2].sel(models=gmodels), "models"), ca.cal_rMME(IndR_his_NC_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*", ec="black")
xyregress = stats.linregress(IndR_his_EAhigh_regress[2].sel(models=gmodels).data,IndR_his_NC_regress[2].sel(models=gmodels).data)
axs[0].line(np.linspace(-0.70,0), xyregress[0]*np.linspace(-0.70,0)+xyregress[1],zorder=0.8,color="sky blue")
axs[0].text(0.5,0.1,s='{:.3f}'.format(xyregress[0]))
#   x-axis title
axs[0].text(-0.90,0.03,s='corr(IndR, EAhigh)')
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
axs[0].format(xlim=(-1.0,1.0), ylim=(-0.6,0.6), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
#   plot the correlation scatter-plot, x:corr(IndR, WNPhigh), y:corr(IndR, NCR)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']
m = axs[0].scatter(IndR_GPCP_WNPhigh_regress[2], IndR_GPCP_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="GPCP", marker="s", color="blue5", ec="black")

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_his_WNPhigh_regress[2].sel(models=mod), IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, ec="black")
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_his_WNPhigh_regress[2].sel(models=mod), IndR_his_NC_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, marker="h", ec="black")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(ca.cal_rMME(IndR_his_WNPhigh_regress[2], "models"), ca.cal_rMME(IndR_his_NC_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^", ec="black")
m = axs[0].scatter(ca.cal_rMME(IndR_his_WNPhigh_regress[2].sel(models=gmodels), "models"), ca.cal_rMME(IndR_his_NC_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*", ec="black")
# xyregress = stats.linregress(IndR_his_WNPhigh_regress[2].sel(models=gmodels).data,IndR_his_NC_regress[2].sel(models=gmodels).data)
# axs[0].line(np.linspace(-0.70,0), xyregress[0]*np.linspace(-0.70,0)+xyregress[1],zorder=0.8,color="sky blue")
# axs[0].text(0.5,0.1,s='{:.3f}'.format(xyregress[0]))
#   x-axis title
axs[0].text(-0.95,0.03,s='corr(IndR, WNPhigh)')
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
axs[0].format(xlim=(-1.0,1.0), ylim=(-0.6,0.6), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
#   plot the India rainfall succession
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=7.0, wspace=4.0, hspace=3.5, outerpad=2.0)
# plot_array = np.reshape(range(1, 4), (1, 3))
# plot_array[-1,-3:] = 0
axs = fig.subplots(ncols=1, nrows=3)
# for gMME
scale = pplt.CutoffScale(2016,np.inf,2062)
axs.format(xlim=(1979,2099),xscale=scale,xlocator=np.append(np.arange(1979,2015,5),np.arange(2064, 2100,5)),ylim=(-3.0,3.0))
# axs = pplt.GridSpec(6,5)
# ax = fig.subplot(axs[0])
# bax = brokenaxes(xlims=((1979, 2014), (2064, 2099)), despine=False, subplot_spec=ax)

m1 = axs[0].plot(np.arange(1979,2015,1), ca.standardize(prehis_India_JJA.sel(time=prehis_India_JJA.time.dt.year>=1979, models=gmodels).mean(dim="models")), color="black", lw=1.2)
axs[0].plot(np.arange(2064,2100,1), ca.standardize(pressp585_India_JJA.sel(time=pressp585_India_JJA.time.dt.year>=2064, models=gmodels).mean(dim="models")), color="black", lw=1.2)
# m2 = axs[0].plot(np.arange(1979,2015,1), ca.standardize(uhis_EA_JJA.sel(time=uhis_EA_JJA.time.dt.year>=1979).mean(dim="models")), color="blue", lw=1.2)
# axs[0].plot(np.arange(2064,2100,1), ca.standardize(ussp585_EA_JJA.sel(time=ussp585_EA_JJA.time.dt.year>=2064).mean(dim="models")), color="blue", lw=1.2)
axs[0].plot(np.arange(1979,2015,1),np.zeros(36),lw=1.2,color="grey7")
axs[0].plot(np.arange(2064,2100,1),np.zeros(36),lw=1.2,color="grey7")
axs[0].fill_between([2014,2064],-3,3,color="grey7", alpha=0.1)
axs[0].format(ltitle="gMME", rtitle="Total India")
# ====================================
m1 = axs[1].plot(np.arange(1979,2015,1), ca.standardize(prehis_nIndia_JJA.sel(time=prehis_nIndia_JJA.time.dt.year>=1979, models=gmodels).mean(dim="models")), color="black", lw=1.2)
axs[1].plot(np.arange(2064,2100,1), ca.standardize(pressp585_nIndia_JJA.sel(time=pressp585_nIndia_JJA.time.dt.year>=2064, models=gmodels).mean(dim="models")), color="black", lw=1.2)
# m2 = axs[1].plot(np.arange(1979,2015,1), ca.standardize(uhis_EA_JJA.sel(time=uhis_EA_JJA.time.dt.year>=1979).mean(dim="models")), color="blue", lw=1.2)
# axs[1].plot(np.arange(2064,2100,1), ca.standardize(ussp585_EA_JJA.sel(time=ussp585_EA_JJA.time.dt.year>=2064).mean(dim="models")), color="blue", lw=1.2)
axs[1].plot(np.arange(1979,2015,1),np.zeros(36),lw=1.2,color="grey7")
axs[1].plot(np.arange(2064,2100,1),np.zeros(36),lw=1.2,color="grey7")
axs[1].fill_between([2014,2064],-3,3,color="grey7", alpha=0.1)
axs[1].format(ltitle="gMME", rtitle="North India")
# =================================
m1 = axs[2].plot(np.arange(1979,2015,1), ca.standardize(prehis_wIndia_JJA.sel(time=prehis_wIndia_JJA.time.dt.year>=1979, models=gmodels).mean(dim="models")), color="black", lw=1.2)
axs[2].plot(np.arange(2064,2100,1), ca.standardize(pressp585_wIndia_JJA.sel(time=pressp585_wIndia_JJA.time.dt.year>=2064, models=gmodels).mean(dim="models")), color="black", lw=1.2)
# m2 = axs[2].plot(np.arange(1979,2015,1), ca.standardize(uhis_EA_JJA.sel(time=uhis_EA_JJA.time.dt.year>=1979).mean(dim="models")), color="blue", lw=1.2)
# axs[2].plot(np.arange(2064,2100,1), ca.standardize(ussp585_EA_JJA.sel(time=ussp585_EA_JJA.time.dt.year>=2064).mean(dim="models")), color="blue", lw=1.2)
axs[2].plot(np.arange(1979,2015,1),np.zeros(36),lw=1.0,color="grey7")
axs[2].plot(np.arange(2064,2100,1),np.zeros(36),lw=1.0,color="grey7")
axs[2].fill_between([2014,2064],-3,3,color="grey7", alpha=0.1)
axs[2].format(ltitle="gMME", rtitle="West India")

fig.format(abc="(a)", abcloc="l", suptitle="IndR")
# %%
#   plot the 200hPa circulation mme result in 1979-2014, 2064-2099, diff
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    # plot_array = np.reshape(range(1, 31), (6, 5))
    # plot_array[-1,-1] = 0
    axs = fig.subplots(ncols=3, nrows=1, proj=proj)

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
        IndR_his_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_ens.sel(level=lev), axs[0], n, np.where(IndR_his_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[0].quiver(
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

    m = axs[0].quiver(
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

    qk = axs[0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[0].format(
        rtitle="1979-2014", ltitle="MME",
    )
    # ======================================
    con = axs[1].contourf(
        IndR_ssp585_p3_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_slope_ens.sel(level=lev), axs[1], n, np.where(IndR_ssp585_p3_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndR_ssp585_p3_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_ens.sel(level=lev)[::ski, ::ski],
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
        IndR_ssp585_p3_u_slope_ens.sel(level=lev).where(IndR_ssp585_p3_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_ssp585_p3_v_slope_ens.sel(level=lev).where(IndR_ssp585_p3_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        rtitle="2064-2099", ltitle="MME",
    )
    # ======================================
    con = axs[2].contourf(
        IndR_diff_hgt_slope_ens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_diff_hgt_slope_ens.sel(level=lev), axs[2], n, np.where(IndR_diff_hgt_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[2].quiver(
        IndR_diff_u_slope_ens.sel(level=lev)[::ski, ::ski],
        IndR_diff_v_slope_ens.sel(level=lev)[::ski, ::ski],
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
        IndR_diff_u_slope_ens.sel(level=lev).where(IndR_diff_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_diff_v_slope_ens.sel(level=lev).where(IndR_diff_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        rtitle="diff", ltitle="MME",
    )
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   plot the bar-plot of the EA high
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_EAhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_EAhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_EAhigh_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_EAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_EAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_EAhigh_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

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
axs[0].format(ylim=(-2.0e-6,2.0e-6),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and EAhigh")
# %%
#   plot the bar-plot of the WA high
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_WAhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_WAhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_WAhigh_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_WAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_WAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_WAhigh_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

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
axs[0].format(ylim=(-3.0e-6,3.0e-6),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and WAhigh")

# %%
#   plot the bar-plot of the WNP high
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_WNPhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_WNPhigh_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_WNPhigh_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_WNPhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_WNPhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_WNPhigh_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

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
axs[0].format(ylim=(-1.0e-6,1.0e-6),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and WNPhigh")
# %%
#   plot the correlation scatter-plot, x:corr(IndR, WAhigh), y:corr(IndR, EAhigh)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']
m = axs[0].scatter(IndR_GPCP_WAhigh_regress[2], IndR_GPCP_EAhigh_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="obs", marker="s", color="blue5", ec="black")

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_his_WAhigh_regress[2].sel(models=mod), IndR_his_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, ec="black")
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_his_WAhigh_regress[2].sel(models=mod), IndR_his_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, marker="h", ec="black")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(ca.cal_rMME(IndR_his_WAhigh_regress[2], "models"), ca.cal_rMME(IndR_his_EAhigh_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^", ec="black")
m = axs[0].scatter(ca.cal_rMME(IndR_his_WAhigh_regress[2].sel(models=gmodels), "models"), ca.cal_rMME(IndR_his_EAhigh_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*", ec="black")
# xyregress = stats.linregress(IndR_his_WAhigh_regress[2].sel(models=gmodels).data,IndR_his_EAhigh_regress[2].sel(models=gmodels).data)
# axs[0].line(np.linspace(-0.70,0), xyregress[0]*np.linspace(-0.70,0)+xyregress[1],zorder=0.8,color="sky blue")
# axs[0].text(0.5,0.1,s='{:.3f}'.format(xyregress[0]))
#   x-axis title
axs[0].text(-0.95,0.03,s='corr(IndR, WAhigh)')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, EAhigh)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.0), ylim=(-1.0,1.0), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
#   plot the correlation scatter-plot, x:corr(IndR, WAhigh), y:corr(IndR, EAhigh)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']
# m = axs[0].scatter(IndR_GPCP_WAhigh_regress[2], IndR_GPCP_EAhigh_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="obs", marker="s", color="blue5", ec="black")

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_ssp585_p3_WAhigh_regress[2].sel(models=mod), IndR_ssp585_p3_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, ec="black")
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_ssp585_p3_WAhigh_regress[2].sel(models=mod), IndR_ssp585_p3_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, marker="h", ec="black")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(ca.cal_rMME(IndR_ssp585_p3_WAhigh_regress[2], "models"), ca.cal_rMME(IndR_ssp585_p3_EAhigh_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^", ec="black")
m = axs[0].scatter(ca.cal_rMME(IndR_ssp585_p3_WAhigh_regress[2].sel(models=gmodels), "models"), ca.cal_rMME(IndR_ssp585_p3_EAhigh_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*", ec="black")
# xyregress = stats.linregress(IndR_ssp585_p3_WAhigh_regress[2].sel(models=gmodels).data,IndR_ssp585_p3_EAhigh_regress[2].sel(models=gmodels).data)
# axs[0].line(np.linspace(-0.70,0), xyregress[0]*np.linspace(-0.70,0)+xyregress[1],zorder=0.8,color="sky blue")
# axs[0].text(0.5,0.1,s='{:.3f}'.format(xyregress[0]))
#   x-axis title
axs[0].text(-0.95,0.03,s='corr(IndR, WAhigh)')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, EAhigh)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.0), ylim=(-1.0,1.0), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="ssp585_p3 Corr Coeff. with IndR")
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
plot_array = np.reshape(range(1, 4), (3, 1))
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
            x0 = EAhigh_W
            y0 = EAhigh_S
            width = EAhigh_E-EAhigh_W
            height = EAhigh_N-EAhigh_S
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
            x0 = WAhigh_W
            y0 = WAhigh_S
            width = WAhigh_E-WAhigh_W
            height = WAhigh_N-WAhigh_S
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
    elif lev == 850.0:
        for ax in axs[num_lev, :]:
            x0 = WNPhigh_W
            y0 = WNPhigh_S
            width = WNPhigh_E-WNPhigh_W
            height = WNPhigh_N-WNPhigh_S
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-")
    # ======================================
    con = axs[num_lev, 0].contourf(
        IndR_his_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_gens.sel(level=lev), axs[num_lev, 0], n, np.where(IndR_his_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[num_lev, 0].quiver(
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

    m = axs[num_lev, 0].quiver(
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

    qk = axs[num_lev, 0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev, 0].format(
        rtitle="1979-2014 {:.0f}hPa".format(lev), ltitle="gMME",
    )
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="hgt&U reg IndR".format(lev))
# %%
#   plot the correlation scatter-plot, x:corr(IndR, WAhigh), y:corr(IndR, EAhigh)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']
# m = axs[0].scatter(IndR_GPCP_WAhigh_regress[2], IndR_GPCP_EAhigh_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="obs", marker="s", color="blue5", ec="black")

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_his_WNPhigh_regress[2].sel(models=mod), IndR_his_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, ec="black")
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_his_WNPhigh_regress[2].sel(models=mod), IndR_his_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, marker="h", ec="black")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(ca.cal_rMME(IndR_his_WNPhigh_regress[2], "models"), ca.cal_rMME(IndR_his_EAhigh_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^", ec="black")
m = axs[0].scatter(ca.cal_rMME(IndR_his_WNPhigh_regress[2].sel(models=gmodels), "models"), ca.cal_rMME(IndR_his_EAhigh_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*", ec="black")
# xyregress = stats.linregress(IndR_his_WNPhigh_regress[2].sel(models=gmodels).data,IndR_his_EAhigh_regress[2].sel(models=gmodels).data)
# axs[0].line(np.linspace(-0.70,0), xyregress[0]*np.linspace(-0.70,0)+xyregress[1],zorder=0.8,color="sky blue")
# axs[0].text(0.5,0.1,s='{:.3f}'.format(xyregress[0]))
#   x-axis title
axs[0].text(-0.95,0.03,s='corr(IndR, WNPhigh)')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, EAhigh)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.0), ylim=(-1.0,1.0), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="his Corr Coeff. with IndR")
# %%
#   plot the correlation scatter-plot, x:corr(IndR, WAhigh), y:corr(IndR, EAhigh)
fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycle = pplt.Cycle('blues', 'acton', 'oranges', 'greens', 28, left=0.1)
# cycle = pplt.Cycle('538', 'Vlag' , 15, left=0.1)
# m = axs[0].scatter(IndR_CRU_SC_regress[2], IndR_CRU_NC_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="CRU", marker="s")
pmodels=['ACCESS-CM2','BCC-CSM2-MR','CESM2','CNRM-CM6-1','CNRM-ESM2-1','CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3','GFDL-CM4','HadGEM3-GC31-LL','INM-CM5-0','IPSL-CM6A-LR','KACE-1-0-G','MIROC-ES2L','MIROC6','MPI-ESM1-2-HR','NESM3','NorESM2-LM','TaiESM1']
# m = axs[0].scatter(IndR_GPCP_WAhigh_regress[2], IndR_GPCP_EAhigh_regress[2], cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="obs", marker="s", color="blue5", ec="black")

for num_models, mod in enumerate(pmodels):
    m = axs[0].scatter(IndR_ssp585_p3_WNPhigh_regress[2].sel(models=mod), IndR_ssp585_p3_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, ec="black")
    
for num_models, mod in enumerate(gmodels):
    m = axs[0].scatter(IndR_ssp585_p3_WNPhigh_regress[2].sel(models=mod), IndR_ssp585_p3_EAhigh_regress[2].sel(models=mod), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels=mod, marker="h", ec="black")
# fig.legend(loc="bottom", labels=models)
# axs[0].axhline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axhline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
# axs[0].axvline(-ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
m = axs[0].scatter(ca.cal_rMME(IndR_ssp585_p3_WNPhigh_regress[2], "models"), ca.cal_rMME(IndR_ssp585_p3_EAhigh_regress[2],"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="MME", marker="^", ec="black")
m = axs[0].scatter(ca.cal_rMME(IndR_ssp585_p3_WNPhigh_regress[2].sel(models=gmodels), "models"), ca.cal_rMME(IndR_ssp585_p3_EAhigh_regress[2].sel(models=gmodels),"models"), cycle=cycle, legend='b', legend_kw={"ncols":4}, labels="gMME", marker="*", ec="black")
# xyregress = stats.linregress(IndR_ssp585_p3_WNPhigh_regress[2].sel(models=gmodels).data,IndR_ssp585_p3_EAhigh_regress[2].sel(models=gmodels).data)
# axs[0].line(np.linspace(-0.70,0), xyregress[0]*np.linspace(-0.70,0)+xyregress[1],zorder=0.8,color="sky blue")
# axs[0].text(0.5,0.1,s='{:.3f}'.format(xyregress[0]))
#   x-axis title
axs[0].text(-0.95,0.03,s='corr(IndR, WNPhigh)')
#   y-axis title
axs[0].text(0.03,-0.55,s='corr(IndR, EAhigh)')

axs[0].hlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.9, 36), -ca.cal_rlim1(0.9, 36),ca.cal_rlim1(0.9, 36), lw=1.2, color="grey7", ls="--")

axs[0].hlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].hlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].vlines(-ca.cal_rlim1(0.95, 36), -ca.cal_rlim1(0.95, 36),ca.cal_rlim1(0.95, 36), lw=1.2, color="grey7", ls="--")
axs[0].format(xlim=(-1.0,1.0), ylim=(-1.0,1.0), xloc="zero", yloc="zero", grid=False, xlabel="", ylabel="", ytickloc="both", xtickloc="both", suptitle="ssp585_p3 Corr Coeff. with IndR")
# %%
#   plot the bar-plot of the IndR related EAM index for historical and ssp585_p3 (reg coeff.)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_EAM_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_EAM_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_EAM_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_EAM_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_EAM_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_EAM_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

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
axs[0].format(ylim=(-3.0,3.0),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and EAM")
# %%
#   plot the bar-plot of the IndR related IWF index for historical and ssp585_p3 (reg coeff.)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_IWF_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_IWF_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_IWF_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_IWF_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_IWF_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_IWF_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

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
axs[0].format(ylim=(-1.0e-6,1.0e-6),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and IWF")
# %%
#   calculate the pcc between precipitation
lat_range = lat[(lat>=15) & (lat<=40.0)]
lon_range = lon[(lon>=90) & (lon<=130.0)]
precip_pcc_list = []
for num_mod, mod in enumerate(models):
    precip_pcc = ca.cal_pcc(pre_AIR_India_pre_slope.sel(lat=lat_range, lon=lon_range), pre_his_India_pre_slope.sel(models=mod, lat=lat_range, lon=lon_range))
    precip_pcc_list.append(precip_pcc)
# %%
#   plot the IndR pcc, NCR pcc and corr(IndR, NCR) in different models
#   first the IndR pcc is divided into two terms: 200hgt pcc and 850hgt pcc

fig = pplt.figure(span=False, share=False, refheight=4.0, refwidth=8.0, wspace=4.0, hspace=3.5, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=1)
cycles = pplt.Cycle("tab10", 2)
axs[0].line(models_array, NCR_200hgt_pcc[:-2], zorder=1, cycle=cycles, label="NCR_200", marker="o")
axs[0].line(models_array, IndR_200hgt_pcc[:-2], zorder=1, cycle=cycles, label="IndR_200", marker="o")
axs[0].line(models_array, NCR_850hgt_pcc[:-2], zorder=1, cycle=cycles, label="NCR_850", marker="o", ls="--")
axs[0].line(models_array, IndR_850hgt_pcc[:-2], zorder=1, cycle=cycles, label="IndR_850", marker="o", ls="--")

# axs[0].line(models_array, ca.cal_rdiff(IndR_his_NC_regress[2], IndR_GPCP_NC_regress[2]), zorder=1, color="jade", marker="*", label="corr_his-corr_GPCP")

axs[0].line(models_array, precip_pcc_list, zorder=1, color="jade", marker="*", label="precip_pcc")

axs.format(xrotation=45, ylim=(-1, 1), ylabel="")
axs[0].legend(loc="ll", ncols=2)
fig.format(suptitle="pcc & corr(IndR, NCR)")
# %%
#   pick up the positive and negative corr(IndR, NCR) models
pcorr_models=['ACCESS-CM2', 'CAMS-CSM1-0', 'CESM2', 'CESM2-WACCM', 'CMCC-ESM2', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-g3', 'GFDL-CM4', 'HadGEM3-GC31-LL', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC6', 'MIROC-ES2L', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'TaiESM1', 'UKESM1-0-LL']
ncorr_models=["BCC-CSM2-MR", "CanESM5", "MPI-ESM1-2-HR"]

#   calculate the ensemble mean of positive corr(IndR, NCR) models
IndR_his_hgt_slope_pens = IndR_his_hgt_slope.sel(models=pcorr_models).mean(dim="models", skipna=True)
IndR_his_u_slope_pens = IndR_his_u_slope.sel(models=pcorr_models).mean(dim="models", skipna=True)
IndR_his_v_slope_pens = IndR_his_v_slope.sel(models=pcorr_models).mean(dim="models", skipna=True)

IndR_ssp585_p3_hgt_slope_pens = IndR_ssp585_p3_hgt_slope.sel(models=pcorr_models).mean(dim="models", skipna=True)
IndR_ssp585_p3_u_slope_pens = IndR_ssp585_p3_u_slope.sel(models=pcorr_models).mean(dim="models", skipna=True)
IndR_ssp585_p3_v_slope_pens = IndR_ssp585_p3_v_slope.sel(models=pcorr_models).mean(dim="models", skipna=True)

IndR_his_hgt_slope_pens_mask = xr.where((ca.MME_reg_mask(IndR_his_hgt_slope_pens, IndR_his_hgt_slope.sel(models=pcorr_models).std(dim="models", skipna=True), len(IndR_his_hgt_slope.sel(models=pcorr_models).coords["models"]), True) + ca.cal_mmemask(IndR_his_hgt_slope.sel(models=pcorr_models))) >= 2.0, 1.0, 0.0)
IndR_his_u_slope_pens_mask = xr.where((ca.MME_reg_mask(IndR_his_u_slope_pens, IndR_his_u_slope.sel(models=pcorr_models).std(dim="models", skipna=True), len(IndR_his_u_slope.sel(models=pcorr_models).coords["models"]), True) + ca.cal_mmemask(IndR_his_u_slope.sel(models=pcorr_models))) >= 2.0, 1.0, 0.0)
IndR_his_v_slope_pens_mask = xr.where((ca.MME_reg_mask(IndR_his_v_slope_pens, IndR_his_v_slope.sel(models=pcorr_models).std(dim="models", skipna=True), len(IndR_his_v_slope.sel(models=pcorr_models).coords["models"]), True) + ca.cal_mmemask(IndR_his_v_slope.sel(models=pcorr_models))) >= 2.0, 1.0, 0.0)
IndR_his_wind_slope_pens_mask = ca.wind_check(
    xr.where(IndR_his_u_slope_pens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_pens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_u_slope_pens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_his_v_slope_pens_mask > 0.0, 1.0, 0.0),
)

IndR_ssp585_p3_hgt_slope_pens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_hgt_slope_pens, IndR_ssp585_p3_hgt_slope.sel(models=pcorr_models).std(dim="models", skipna=True), len(IndR_ssp585_p3_hgt_slope.sel(models=pcorr_models).coords["models"]), True) + ca.cal_mmemask(IndR_ssp585_p3_hgt_slope.sel(models=pcorr_models))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_u_slope_pens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_u_slope_pens, IndR_ssp585_p3_u_slope.sel(models=pcorr_models).std(dim="models", skipna=True), len(IndR_ssp585_p3_u_slope.sel(models=pcorr_models).coords["models"]), True) + ca.cal_mmemask(IndR_ssp585_p3_u_slope.sel(models=pcorr_models))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_v_slope_pens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_v_slope_pens, IndR_ssp585_p3_v_slope.sel(models=pcorr_models).std(dim="models", skipna=True), len(IndR_ssp585_p3_v_slope.sel(models=pcorr_models).coords["models"]), True) + ca.cal_mmemask(IndR_ssp585_p3_v_slope.sel(models=pcorr_models))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_wind_slope_pens_mask = ca.wind_check(
    xr.where(IndR_ssp585_p3_u_slope_pens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_pens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_u_slope_pens_mask > 0.0, 1.0, 0.0),
    xr.where(IndR_ssp585_p3_v_slope_pens_mask > 0.0, 1.0, 0.0),
)

# %%
#   plot the MME and seperate models results
startlevel=[-15, -8, -6]
spacinglevel=[0.75, 0.4, 0.3]
scalelevel=[0.23, 0.17, 0.14]
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    pplt.rc.grid = False
    pplt.rc.reso = "lo"
    cl = 0  # 设置地图投影的中心纬度
    proj = pplt.PlateCarree(central_longitude=cl)

    fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
    plot_array = np.reshape(range(1, 26), (5, 5))
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
        IndR_his_hgt_slope_pens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_pens.sel(level=lev), axs[1], n, np.where(IndR_his_hgt_slope_pens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 3.0,
    )
    axs[1].quiver(
        IndR_his_u_slope_pens.sel(level=lev)[::ski, ::ski],
        IndR_his_v_slope_pens.sel(level=lev)[::ski, ::ski],
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
        IndR_his_u_slope_pens.sel(level=lev).where(IndR_his_wind_slope_pens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_pens.sel(level=lev).where(IndR_his_wind_slope_pens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
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
        rtitle="1979-2014", ltitle="pMME",
    )
    # ======================================
    for num_mod, mod in enumerate(pcorr_models):
        con = axs[num_mod+2].contourf(
            IndR_his_hgt_slope.sel(models=mod,level=lev),
            cmap="ColdHot",
            cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
            levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
            zorder=0.8,
            extend="both"
        )
        sepl.plt_sig(
            IndR_his_hgt_slope.sel(models=mod,level=lev), axs[num_mod+2], n, np.where(IndR_his_hgt_pvalue.sel(models=mod,level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
        )
        axs[num_mod+2].quiver(
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

        m = axs[num_mod+2].quiver(
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

        qk = axs[num_mod+2].quiverkey(
            m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
        )
        axs[num_mod+2].format(
            rtitle="1979-2014", ltitle="{}".format(mod),
        )
    # ======================================
    fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
    fig.format(abc="(a)", abcloc="l", suptitle="{:.0f}hPa hgt&U reg IndR".format(lev))
# %%
#   calculate the circulation result regress onto corr and slope between models

(
    IndR_NC_intermodels_rvalue_his_hgt_slope,
    IndR_NC_intermodels_rvalue_his_hgt_intercept,
    IndR_NC_intermodels_rvalue_his_hgt_rvalue,
    IndR_NC_intermodels_rvalue_his_hgt_pvalue,
    IndR_NC_intermodels_rvalue_his_hgt_hypothesis,
) = ca.dim_linregress(IndR_his_NC_regress[2].sel(models=pcorr_models), IndR_his_hgt_rvalue.sel(models=pcorr_models), input_core_dims="models")

(
    IndR_NC_intermodels_rvalue_his_u_slope,
    IndR_NC_intermodels_rvalue_his_u_intercept,
    IndR_NC_intermodels_rvalue_his_u_rvalue,
    IndR_NC_intermodels_rvalue_his_u_pvalue,
    IndR_NC_intermodels_rvalue_his_u_hypothesis,
) = ca.dim_linregress(IndR_his_NC_regress[2].sel(models=pcorr_models), IndR_his_u_rvalue.sel(models=pcorr_models), input_core_dims="models")

(
    IndR_NC_intermodels_rvalue_his_v_slope,
    IndR_NC_intermodels_rvalue_his_v_intercept,
    IndR_NC_intermodels_rvalue_his_v_rvalue,
    IndR_NC_intermodels_rvalue_his_v_pvalue,
    IndR_NC_intermodels_rvalue_his_v_hypothesis,
) = ca.dim_linregress(IndR_his_NC_regress[2].sel(models=pcorr_models), IndR_his_v_rvalue.sel(models=pcorr_models), input_core_dims="models")

IndR_NC_intermodels_rvalue_his_wind_mask = ca.wind_check(
    xr.where(IndR_NC_intermodels_rvalue_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_NC_intermodels_rvalue_his_v_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_NC_intermodels_rvalue_his_u_pvalue <= 0.05, 1.0, 0.0),
    xr.where(IndR_NC_intermodels_rvalue_his_v_pvalue <= 0.05, 1.0, 0.0),
)
# %%
#   plot the intermodels regression results
startlevel=[-1.0, -1.0, -1.0]
spacinglevel=[0.1, 0.1, 0.1]
scalelevel=[0.17, 0.17, 0.17]
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 4), (3, 1))
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
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    con = axs[num_lev].contourf(
        IndR_NC_intermodels_rvalue_his_hgt_rvalue.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_NC_intermodels_rvalue_his_hgt_rvalue.sel(level=lev), axs[num_lev], n, np.where(IndR_NC_intermodels_rvalue_his_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_lev].quiver(
        IndR_NC_intermodels_rvalue_his_u_rvalue.sel(level=lev)[::ski, ::ski],
        IndR_NC_intermodels_rvalue_his_v_rvalue.sel(level=lev)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="grey6",
    )

    m = axs[num_lev].quiver(
        IndR_NC_intermodels_rvalue_his_u_rvalue.sel(level=lev).where(IndR_NC_intermodels_rvalue_his_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_NC_intermodels_rvalue_his_v_rvalue.sel(level=lev).where(IndR_NC_intermodels_rvalue_his_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.05, fontproperties={"size": 5}, zorder=3.1,
    )
    axs[num_lev].format(
        rtitle="{:.0f}hPa".format(lev), ltitle="historical",
    )
    
    # ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="(hgt&U reg IndR) reg corr(IndR,NCR)")
# %%
# plot the bar plot about the corr(IndR, LKY)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_LKY_regress[0].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_LKY_regress[0].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_LKY_slope.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_LKY_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_LKY_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_LKY_slope.sel(models=gmodels).mean(dim="models", skipna=True).data

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
axs[0].format(ylim=(-1e-6,1e-6),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and LKY")

# plot the bar plot about the corr(IndR, LKY)
plot_data = np.zeros((7,3))
plot_data[:-1,0] = IndR_his_LKY_regress[2].sel(models=gmodels).data
plot_data[:-1,1] = IndR_ssp585_p3_LKY_regress[2].sel(models=gmodels).data
plot_data[:-1,2] = IndR_diff_LKY_rvalue.sel(models=gmodels).data
plot_data[-1,0] = IndR_his_LKY_regress[2].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_LKY_regress[2].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_LKY_rvalue.sel(models=gmodels).mean(dim="models", skipna=True).data

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
axs[0].format(ylim=(-1,1),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Corr. Coeff. IndR and LKY")
# %%
#   plot the all-models bar-plots for corr(IndR, LKY)
plot_data = np.zeros((27,3))
plot_data[:-1,0] = IndR_his_LKY_regress[0].data
plot_data[:-1,1] = IndR_ssp585_p3_LKY_regress[0].data
plot_data[:-1,2] = IndR_diff_LKY_slope.data
plot_data[-1,0] = IndR_his_LKY_regress[0].mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_LKY_regress[0].mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_LKY_slope.mean(dim="models", skipna=True).data

label_models = list(models.data)
label_models.append("MME")

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
axs[0].format(ylim=(-1e-6,1e-6),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Reg. Coeff. IndR and LKY")

# plot the bar plot about the corr(IndR, LKY)
plot_data = np.zeros((27,3))
plot_data[:-1,0] = IndR_his_LKY_regress[2].data
plot_data[:-1,1] = IndR_ssp585_p3_LKY_regress[2].data
plot_data[:-1,2] = IndR_diff_LKY_rvalue.data
plot_data[-1,0] = IndR_his_LKY_regress[2].mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_LKY_regress[2].mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_LKY_rvalue.mean(dim="models", skipna=True).data

label_models = list(models.data)
label_models.append("MME")

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
axs[0].format(ylim=(-1,1),xlocator=np.arange(0,27), xtickminor=False, ytickminor=False, grid=False, xrotation=45, xticklabelsize=12, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8")
# ax.outline_patch.set_linewidth(1.0)
fig.format(suptitle="Corr. Coeff. IndR and LKY")
# %%
#   plot the SST regress onto the IndR in historical
#   corr. coeffs.
startlevel=-1.0
spacinglevel=0.1

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_Had_sst_rvalue,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
)
sepl.plt_sig(
    IndR_Had_sst_rvalue, axs[0], n, np.where(IndR_Had_sst_pvalue[::n, ::n] <= 0.05), "bright purple", 3.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="HadISST & AIR",
)
# ======================================
con = axs[1].contourf(
    IndR_his_sst_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
)
sepl.plt_sig(
    IndR_his_sst_rvalue_ens, axs[1], n, np.where(IndR_his_sst_rvalue_ens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[1].format(
    rtitle="1979-2014", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models_array):
    con = axs[num_mod+2].contourf(
        IndR_his_sst_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_his_sst_rvalue.sel(models=mod), axs[num_mod+2], n, np.where(IndR_his_sst_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")

# ======================================
#   reg. coeffs.
startlevel=-6e-1
spacinglevel=0.06

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-2:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_Had_sst_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_Had_sst_slope, axs[0], n, np.where(IndR_Had_sst_pvalue[::n, ::n] <= 0.05), "bright purple", 3.0,
)

axs[0].format(
    rtitle="1979-2014", ltitle="HadISST & AIR",
)
# ======================================
con = axs[1].contourf(
    IndR_his_sst_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_his_sst_slope_ens, axs[1], n, np.where(IndR_his_sst_slope_ens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[1].format(
    rtitle="1979-2014", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models_array):
    con = axs[num_mod+2].contourf(
        IndR_his_sst_slope.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_sst_slope.sel(models=mod), axs[num_mod+2], n, np.where(IndR_his_sst_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+2].format(
        rtitle="1979-2014", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")
# %%
#   plot the SST regress onto the IndR in ssp585_p3
#   corr. coeffs.
startlevel=-1.0
spacinglevel=0.1

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14

# ======================================
con = axs[0].contourf(
    IndR_ssp585_p3_sst_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
)
sepl.plt_sig(
    IndR_ssp585_p3_sst_rvalue_ens, axs[0], n, np.where(IndR_ssp585_p3_sst_rvalue_ens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[0].format(
    rtitle="2064-2099", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models_array):
    con = axs[num_mod+1].contourf(
        IndR_ssp585_p3_sst_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
    )
    sepl.plt_sig(
        IndR_ssp585_p3_sst_rvalue.sel(models=mod), axs[num_mod+1], n, np.where(IndR_ssp585_p3_sst_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")

# ======================================
#   reg. coeffs.
startlevel=-6.0e-1
spacinglevel=0.06

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_ssp585_p3_sst_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_ssp585_p3_sst_slope_ens, axs[0], n, np.where(IndR_ssp585_p3_sst_slope_ens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[0].format(
    rtitle="2064-2099", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models_array):
    con = axs[num_mod+1].contourf(
        IndR_ssp585_p3_sst_slope.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_sst_slope.sel(models=mod), axs[num_mod+1], n, np.where(IndR_ssp585_p3_sst_pvalue.sel(models=mod)[::n, ::n] <= 0.05), "bright purple", 3.0,
    )
    axs[num_mod+1].format(
        rtitle="2064-2099", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")
# %%
#   plot the SST regress onto the IndR in diff
#   corr. coeffs.
startlevel=-1.0
spacinglevel=0.1

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14

# ======================================
con = axs[0].contourf(
    IndR_diff_sst_rvalue_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
)
sepl.plt_sig(
    IndR_diff_sst_rvalue_ens, axs[0], n, np.where(IndR_diff_sst_ens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[0].format(
    rtitle="diff", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models_array):
    con = axs[num_mod+1].contourf(
        IndR_diff_sst_rvalue.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
    )
    axs[num_mod+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")

# ======================================
#   reg. coeffs.
startlevel=-6.0e-1
spacinglevel=0.06

pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 31), (5, 6))
plot_array[-1,-3:] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 360])  # 设置纬度刻度
yticks = np.arange(-60, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents)
# ===================================================
ski = 2
n = 2
w, h = 0.12, 0.14
# ======================================
con = axs[0].contourf(
    IndR_diff_sst_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_diff_sst_slope_ens, axs[0], n, np.where(IndR_diff_sst_ens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[0].format(
    rtitle="diff", ltitle="MME",
)
# ======================================
for num_mod, mod in enumerate(models_array):
    con = axs[num_mod+1].contourf(
        IndR_diff_sst_slope.sel(models=mod),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
        zorder=0.8,
        extend="both"
    )
    axs[num_mod+1].format(
        rtitle="diff", ltitle="{}".format(mod),
    )
# ======================================
fig.colorbar(con, loc="b", width=0.13, length=0.7, label="")
fig.format(abc="(a)", abcloc="l", suptitle="SST reg IndR")
# %%
