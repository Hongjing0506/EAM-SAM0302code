'''
Author: ChenHJ
Date: 2022-07-07 15:42:39
LastEditors: ChenHJ
LastEditTime: 2022-07-08 15:42:23
FilePath: /chenhj/0302code/final_figures.py
Aim: This file is to plot final figures in paper.
There are 10 figures in paper.
1) precipitation regress onto SASMR, observation and MME, 1979-2014
2) circulation regress onto SASMR, observation and MME, 1979-2014, 200/500/850hPa
3) taylor-diagram including 200hPa pcc and 850hPa pcc
4) precipitation regress onto SASMR, gMME, 1979-2014 & 2064-2099 & diff
5) circulation regress onto SASMR, gMME, 1979-2014 & 2064-2099 & diff, 200/500/850hPa
6) bar plots that show the WAAC, EAAC and WNPAC in all 26 models, gMME and MME in the period of 1979-2014 & 2064-2099 & diff
7) Sea surface temperature regress onto SASMR, observation and gMME, 1979-2014 & 2064-2099 & diff
8) without trend: precipitation regress onto SASMR, gMME, 1979-2014 & 2064-2099 & diff
9) without trend: circulation regress onto SASMR, gMME, 1979-2014 & 2064-2099 & diff, 200/500/850hPa
10) bar plots that show the WAAC, EAAC and WNPAC in all 26 models, gMME and MME in the period of 1979-2014 & 2064-2099 & diff
Mission: 
'''
# %%
# thie part is for the nondetrend plots
from mailbox import _PartialFile
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

#   read data from observation and reanalysis data

fpreGPCP = xr.open_dataset(
    "/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/GPCP_r144x72_197901-201412.nc"
)
preGPCP = fpreGPCP["precip"]
preGPCP_JJA = ca.p_time(preGPCP, 6, 8, True)

preAIR = xr.open_dataarray("/home/ys17-23/Extension/All_India_Rainfall_index/AIR_mmperday.nc")
preAIR_JJA = ca.p_time(preAIR, 6, 8, True)
preAIR_JJA = preAIR_JJA.sel(time=(preAIR_JJA.time.dt.year>=1979) & (preAIR_JJA.time.dt.year <=2014))

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

ftERA5 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/obs/temp_mon_r144x72_195001-201412.nc")
tERA5 = ftERA5["t"].sel(time=ftERA5["time"].dt.year>=1979)
tERA5_ver_JJA = ca.p_time(tERA5, 6, 8, True)

# read the data from CMIP6 historical experiment
fprehis = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/pr_historical_r144x72_195001-201412.nc")
prehis_JJA = fprehis["pr"].sel(time=fprehis["time"].dt.year>=1979)
prehis_JJA.attrs["units"] = "mm/day"
prehis_JJA.attrs["standard_name"] = "precipitation"

fhgthis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/zg_historical_r144x72_195001-201412.nc")
hgthis_ver_JJA = fhgthis_ver_JJA["zg"].sel(time=fhgthis_ver_JJA["time"].dt.year>=1979)
hgthis_ver_JJA = hgthis_ver_JJA - hgthis_ver_JJA.mean(dim="lon", skipna=True)

fuhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ua_historical_r144x72_195001-201412.nc")
uhis_ver_JJA = fuhis_ver_JJA["ua"].sel(time=fuhis_ver_JJA["time"].dt.year>=1979)

fvhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/va_historical_r144x72_195001-201412.nc")
vhis_ver_JJA = fvhis_ver_JJA["va"].sel(time=fvhis_ver_JJA["time"].dt.year>=1979)

fwhis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/wap_historical_r144x72_195001-201412.nc") 
whis_ver_JJA = fwhis_ver_JJA["wap"].sel(time=fwhis_ver_JJA["time"].dt.year>=1979)

fssthis_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/tos_historical_r144x72_195001-201412.nc")
ssthis_JJA = fssthis_JJA["tos"].sel(time=fssthis_JJA["time"].dt.year>=1979)

fthis_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/ta_historical_r144x72_195001-201412.nc")
this_ver_JJA = fthis_ver_JJA["ta"].sel(time=fthis_ver_JJA["time"].dt.year>=1979)

# read the data from CMIP6 ssp585 experiment and read the last period of ssp585
fpressp585_p3 = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/pr_ssp585_r144x72_201501-209912.nc")
pressp585_p3_JJA = fpressp585_p3["pr"].sel(time=fpressp585_p3.time.dt.year>=2064)
pressp585_p3_JJA.attrs["units"] = "mm/day"
pressp585_p3_JJA.attrs["standard_name"] = "precipitation"

fhgtssp585_p3_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/zg_ssp585_r144x72_201501-209912.nc")
hgtssp585_p3_ver_JJA = fhgtssp585_p3_ver_JJA["zg"].sel(time=fhgtssp585_p3_ver_JJA.time.dt.year>=2064)
hgtssp585_p3_ver_JJA = hgtssp585_p3_ver_JJA - hgtssp585_p3_ver_JJA.mean(dim="lon", skipna=True)


fussp585_p3_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ua_ssp585_r144x72_201501-209912.nc")
ussp585_p3_ver_JJA = fussp585_p3_ver_JJA["ua"].sel(time=fussp585_p3_ver_JJA.time.dt.year>=2064)

fvssp585_p3_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/va_ssp585_r144x72_201501-209912.nc")
vssp585_p3_ver_JJA = fvssp585_p3_ver_JJA["va"].sel(time=fvssp585_p3_ver_JJA.time.dt.year>=2064)

fwssp585_p3_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/wap_ssp585_r144x72_201501-209912.nc")
wssp585_p3_ver_JJA = fwssp585_p3_ver_JJA["wap"].sel(time=fwssp585_p3_ver_JJA.time.dt.year>=2064)

fsstssp585_p3_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/tos_ssp585_r144x72_201501-209912.nc")
sstssp585_p3_JJA = fsstssp585_p3_JJA["tos"].sel(time=fsstssp585_p3_JJA.time.dt.year>=2064)

ftssp585_p3_ver_JJA = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ta_ssp585_r144x72_201501-209912.nc")
tssp585_p3_ver_JJA = ftssp585_p3_ver_JJA["ta"].sel(time=ftssp585_p3_ver_JJA.time.dt.year>=2064)

# ERA5_EAM = ca.EAM(uERA5_ver_JJA)

# ERA5_IWF = ca.IWF(uERA5_ver_JJA, vERA5_ver_JJA)

# ERA5_LKY = ca.LKY(uERA5_ver_JJA, vERA5_ver_JJA)

# fhis_EAM = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_EAM_index_1950-2014.nc")
# his_EAM = fhis_EAM["EAM"].sel(time=fhis_EAM["time"].dt.year>=1979)

# fssp585_EAM = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_EAM_index_2015-2099.nc")
# ssp585_EAM = fssp585_EAM["EAM"]
# ssp585_p3_EAM = fssp585_EAM["EAM"].sel(time=fssp585_EAM["time"].dt.year>=2064)

# fhis_IWF = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_IWF_index_1950-2014.nc")
# his_IWF = fhis_IWF["IWF"].sel(time=fhis_IWF["time"].dt.year>=1979)

# fssp585_IWF = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_IWF_index_2015-2099.nc")
# ssp585_IWF = fssp585_IWF["IWF"]
# ssp585_p3_IWF = fssp585_IWF["IWF"].sel(time=fssp585_IWF["time"].dt.year>=2064)

#   read the pcc data from the files
# fNCR_200hgt_pcc = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/NCR_200hgt_pcc.nc")
# NCR_200hgt_pcc = fNCR_200hgt_pcc["pcc"]

# fNCR_850hgt_pcc = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/NCR_850hgt_pcc.nc")
# NCR_850hgt_pcc = fNCR_850hgt_pcc["pcc"]
#   read the his_dpg and ssp585_dpg
# his_dsdpg = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/his_dsdpg500-200.nc")
# ssp585_dsdpg = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/ssp585_dsdpg500-200.nc")

# #   calculate the utt in historical and ssp585
# utthis_JJA = (this_ver_JJA.loc[:,:,500.0:200.0,:,:]*his_dsdpg.data).sum(dim="level",skipna=True)
# uttssp585_JJA = (tssp585_ver_JJA.loc[:,:,500.0:200.0,:,:]*ssp585_dsdpg.data).sum(dim="level",skipna=True)
# utthis_JJA.name="utt"
# uttssp585_JJA.name="utt"

#   deal with the time index for CRU and GPCP data
# preCRU_JJA.coords["time"] = prehis_JJA.coords["time"]
preGPCP_JJA.coords["time"] = prehis_JJA.sel(time=prehis_JJA.time.dt.year>=1979).coords["time"]
preAIR_JJA.coords["time"] = prehis_JJA.coords["time"]

models = uhis_ver_JJA.coords["models"]
models_array = models.data

#   pick up the area data
#   calculate the vorticity in ERA5, historical and ssp585
vorERA5_ver_JJA = mpcalc.vorticity(uERA5_ver_JJA.sel(level=200.0), vERA5_ver_JJA.sel(level=200.0))
vorERA5_ver_JJA = vorERA5_ver_JJA.metpy.dequantify()

vorhis_ver_JJA = mpcalc.vorticity(uhis_ver_JJA.sel(level=200.0), vhis_ver_JJA.sel(level=200.0))
vorhis_ver_JJA = vorhis_ver_JJA.metpy.dequantify()

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
pressp585_p3_India_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_India_range, lon=lon_India_range)).mean(dim="lon", skipna=True)

prehis_India_JJA_sum = prehis_JJA.sel(lat=lat_India_range, lon=lon_India_range).sum(dim=["lat", "lon"], skipna=True)
pressp585_p3_India_JJA_sum = pressp585_p3_JJA.sel(lat=lat_India_range, lon=lon_India_range).sum(dim=["lat", "lon"], skipna=True)

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
pressp585_p3_nIndia_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range)).mean(dim="lon", skipna=True)

prehis_nIndia_JJA_sum = prehis_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range).sum(dim=["lat", "lon"], skipna=True)
pressp585_p3_nIndia_JJA_sum = pressp585_p3_JJA.sel(lat=lat_nIndia_range, lon=lon_nIndia_range).sum(dim=["lat", "lon"], skipna=True)

#   calculate the west India peninsula precipitation
wIndia_N = 32.5
# wIndia_N = 30.0
wIndia_S = 8.0
wIndia_W = 70.0
wIndia_E = 77.0
lat_wIndia_range = lat[(lat >= wIndia_S) & (lat <= wIndia_N)]
lon_wIndia_range = lon[(lon >= wIndia_W) & (lon <= wIndia_E)]

prehis_wIndia_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_wIndia_range, lon=lon_wIndia_range)).mean(dim="lon", skipna=True)
pressp585_p3_wIndia_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_wIndia_range, lon=lon_wIndia_range)).mean(dim="lon", skipna=True)

prehis_wIndia_JJA_sum = prehis_JJA.sel(lat=lat_wIndia_range, lon=lon_wIndia_range).sum(dim=["lat", "lon"], skipna=True)
pressp585_p3_wIndia_JJA_sum = pressp585_p3_JJA.sel(lat=lat_wIndia_range, lon=lon_wIndia_range).sum(dim=["lat", "lon"], skipna=True)

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
pressp585_p3_NC_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_NC_range, lon=lon_NC_range)).mean(dim="lon", skipna=True)

#   calculate the precipitation in Southern China
lat_SC_range = lat[(lat>=20.0) & (lat<=27.5)]
lon_SC_range = lon[(lon>=105.0) & (lon<=125.0)]
# preCRU_SC_JJA = ca.cal_lat_weighted_mean(preCRU_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
preGPCP_SC_JJA = ca.cal_lat_weighted_mean(preGPCP_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
prehis_SC_JJA = ca.cal_lat_weighted_mean(prehis_JJA.sel(lat=lat_SC_range, lon=lon_SC_range)).mean(dim="lon", skipna=True)
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
pressp585_p3_SJ_JJA = ca.cal_lat_weighted_mean(pressp585_p3_JJA.sel(lat=lat_SJ_range, lon=lon_SJ_range)).mean(dim="lon", skipna=True)

#   calculate the 200hPa u-wind over the East Asia
lat_EA_range = lat[(lat>=30.0) & (lat<=40.0)]
lon_EA_range = lon[(lon>=100.0) & (lon<=120.0)]

uERA5_EA_JJA = ca.cal_lat_weighted_mean(uERA5_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)
uhis_EA_JJA = ca.cal_lat_weighted_mean(uhis_ver_JJA.sel(lat=lat_EA_range, lon=lon_EA_range, level=200.0)).mean(dim="lon", skipna=True)
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
ussp585_p3_EAhigh_JJA = ussp585_p3_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)

vERA5_EAhigh_JJA = vERA5_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
vhis_EAhigh_JJA = vhis_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)
vssp585_p3_EAhigh_JJA = vssp585_p3_ver_JJA.sel(lat=lat_EAhigh_range, lon=lon_EAhigh_range, level=200.0)

vorERA5_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uERA5_EAhigh_JJA, vERA5_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorhis_EAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uhis_EAhigh_JJA, vhis_EAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
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
ussp585_p3_WAhigh_JJA = ussp585_p3_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)

vERA5_WAhigh_JJA = vERA5_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
vhis_WAhigh_JJA = vhis_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)
vssp585_p3_WAhigh_JJA = vssp585_p3_ver_JJA.sel(lat=lat_WAhigh_range, lon=lon_WAhigh_range, level=200.0)

vorERA5_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uERA5_WAhigh_JJA, vERA5_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorhis_WAhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uhis_WAhigh_JJA, vhis_WAhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
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
ussp585_p3_WNPhigh_JJA = ussp585_p3_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)

vERA5_WNPhigh_JJA = vERA5_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
vhis_WNPhigh_JJA = vhis_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)
vssp585_p3_WNPhigh_JJA = vssp585_p3_ver_JJA.sel(lat=lat_WNPhigh_range, lon=lon_WNPhigh_range, level=850.0)

vorERA5_WNPhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uERA5_WNPhigh_JJA, vERA5_WNPhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorhis_WNPhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(uhis_WNPhigh_JJA, vhis_WNPhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()
vorssp585_p3_WNPhigh_JJA = ca.cal_lat_weighted_mean(mpcalc.vorticity(ussp585_p3_WNPhigh_JJA, vssp585_p3_WNPhigh_JJA)).mean(dim="lon", skipna=True).metpy.dequantify()

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

#   calculate the MME for historical and ssp585_p3
pre_his_India_pre_slope_ens = pre_his_India_pre_slope.mean(dim="models", skipna=True)

pre_his_India_pre_slope_ens_mask = xr.where((ca.MME_reg_mask(pre_his_India_pre_slope_ens, pre_his_India_pre_slope.std(dim="models", skipna=True), len(pre_his_India_pre_slope.coords["models"]), True) + ca.cal_mmemask(pre_his_India_pre_slope)) >= 2.0, 1.0, 0.0)

pre_his_India_pre_rvalue_ens = ca.cal_rMME(pre_his_India_pre_rvalue,"models")

pre_his_India_pre_rvalue_ens_mask = xr.where((ca.MME_reg_mask(pre_his_India_pre_rvalue_ens, pre_his_India_pre_rvalue.std(dim="models", skipna=True), len(pre_his_India_pre_rvalue.coords["models"]), True) + ca.cal_mmemask(pre_his_India_pre_slope)) >= 2.0, 1.0, 0.0)

pre_ssp585_p3_India_pre_slope_ens = pre_ssp585_p3_India_pre_slope.mean(dim="models", skipna=True)

pre_ssp585_p3_India_pre_slope_ens_mask = xr.where((ca.MME_reg_mask(pre_ssp585_p3_India_pre_slope_ens, pre_ssp585_p3_India_pre_slope.std(dim="models", skipna=True), len(pre_ssp585_p3_India_pre_slope.coords["models"]), True) + ca.cal_mmemask(pre_ssp585_p3_India_pre_slope)) >= 2.0, 1.0, 0.0)

pre_ssp585_p3_India_pre_rvalue_ens = ca.cal_rMME(pre_ssp585_p3_India_pre_rvalue,"models")

pre_ssp585_p3_India_pre_rvalue_ens_mask = xr.where((ca.MME_reg_mask(pre_ssp585_p3_India_pre_rvalue_ens, pre_ssp585_p3_India_pre_rvalue.std(dim="models", skipna=True), len(pre_ssp585_p3_India_pre_rvalue.coords["models"]), True) + ca.cal_mmemask(pre_ssp585_p3_India_pre_slope)) >= 2.0, 1.0, 0.0)

#   read the regression data

IndRAIR_ERA5_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndRAIR_ERA5_hgt_regress.nc")
IndRAIR_ERA5_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndRAIR_ERA5_u_regress.nc")
IndRAIR_ERA5_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndRAIR_ERA5_v_regress.nc")

IndR_his_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_hgt_regress.nc")
IndR_his_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_u_regress.nc")
IndR_his_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_v_regress.nc")

IndR_ssp585_p3_hgt_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_hgt_regress.nc")
IndR_ssp585_p3_u_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_u_regress.nc")
IndR_ssp585_p3_v_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_v_regress.nc")

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

#   read the sst regression data
IndR_Had_sst_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_Had_sst_regress.nc")
IndR_his_sst_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/historical/tmp_var/JJA/non_detrend/nIndR_his_sst_regress.nc")
IndR_ssp585_p3_sst_regress = xr.open_dataset("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/nIndR_ssp585_p3_sst_regress.nc")

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

#   calculate the ranking of different models but use the AIR data as observation
lat = prehis_JJA.coords["lat"]
lon = prehis_JJA.coords["lon"]

lon_ranking1_W = 50
lon_ranking1_E = 137.5
lat_ranking1_N = 47.5
lat_ranking1_S = 15.0

lon_ranking2_W = 110
lon_ranking2_E = 137.5
lat_ranking2_N = 37.5
lat_ranking2_S = 15.0
#   for 200hPa
lat_ranking_range1 = lat[(lat>=lat_ranking1_S) & (lat<=lat_ranking1_N)]
lon_ranking_range1 = lon[(lon>=lon_ranking1_W) & (lon<=lon_ranking1_E)]

#   for 850hPa
lat_ranking_range2 = lat[(lat>=lat_ranking2_S) & (lat<=lat_ranking2_N)]
lon_ranking_range2 = lon[(lon>=lon_ranking2_W) & (lon<=lon_ranking2_E)]

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

IndR_his_sst_slope_gens = IndR_his_sst_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_ssp585_p3_sst_slope_gens = IndR_ssp585_p3_sst_slope.sel(models=gmodels).mean(dim="models", skipna=True)


pre_his_India_pre_rvalue_gens = ca.cal_rMME(pre_his_India_pre_rvalue.sel(models=gmodels), "models")
pre_ssp585_p3_India_pre_rvalue_gens = ca.cal_rMME(pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels), "models")

IndR_his_hgt_rvalue_gens = ca.cal_rMME(IndR_his_hgt_rvalue.sel(models=gmodels), "models")
IndR_his_u_rvalue_gens = ca.cal_rMME(IndR_his_u_rvalue.sel(models=gmodels), "models")
IndR_his_v_rvalue_gens = ca.cal_rMME(IndR_his_v_rvalue.sel(models=gmodels), "models")

IndR_ssp585_p3_hgt_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_hgt_rvalue.sel(models=gmodels), "models")
IndR_ssp585_p3_u_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_u_rvalue.sel(models=gmodels), "models")
IndR_ssp585_p3_v_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_v_rvalue.sel(models=gmodels), "models")

IndR_his_sst_rvalue_gens = ca.cal_rMME(IndR_his_sst_rvalue.sel(models=gmodels), "models")
IndR_ssp585_p3_sst_rvalue_gens = ca.cal_rMME(IndR_ssp585_p3_sst_rvalue.sel(models=gmodels), "models")

IndR_his_hgt_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_hgt_slope_gens, IndR_his_hgt_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_hgt_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_u_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_u_slope_gens, IndR_his_u_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_u_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_v_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_v_slope_gens, IndR_his_v_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_v_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_sst_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_sst_slope_gens, IndR_his_sst_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_sst_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_hgt_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_hgt_slope_gens, IndR_ssp585_p3_hgt_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_hgt_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_u_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_u_slope_gens, IndR_ssp585_p3_u_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_u_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_v_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_v_slope_gens, IndR_ssp585_p3_v_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_v_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_sst_slope_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_sst_slope_gens, IndR_ssp585_p3_sst_slope.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_sst_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

pre_his_India_pre_rvalue_gens_mask = xr.where((ca.MME_reg_mask(pre_his_India_pre_rvalue_gens, pre_his_India_pre_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(pre_his_India_pre_rvalue.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
pre_ssp585_p3_India_pre_rvalue_gens_mask = xr.where((ca.MME_reg_mask(pre_ssp585_p3_India_pre_rvalue_gens, pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(pre_ssp585_p3_India_pre_rvalue.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

IndR_his_hgt_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_hgt_rvalue_gens, IndR_his_hgt_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_hgt_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_u_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_u_rvalue_gens, IndR_his_u_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_u_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_v_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_v_rvalue_gens, IndR_his_v_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_v_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_his_sst_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_his_sst_rvalue_gens, IndR_his_sst_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_his_sst_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

IndR_ssp585_p3_hgt_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_hgt_rvalue_gens, IndR_ssp585_p3_hgt_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_hgt_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_u_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_u_rvalue_gens, IndR_ssp585_p3_u_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_u_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_v_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_v_rvalue_gens, IndR_ssp585_p3_v_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_v_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)
IndR_ssp585_p3_sst_rvalue_gens_mask = xr.where((ca.MME_reg_mask(IndR_ssp585_p3_sst_rvalue_gens, IndR_ssp585_p3_sst_rvalue.sel(models=gmodels).std(dim="models", skipna=True), len(gmodels), True) + ca.cal_mmemask(IndR_ssp585_p3_sst_slope.sel(models=gmodels))) >= 2.0, 1.0, 0.0)

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

#   calculate the East Asia westerly jet axis in ERA5, historical, ssp585_p3
wj_area_N = 50.0
wj_area_S = 20.0
wj_area_E = 140.0
wj_area_W = 40.0
lat_wj_range = lat[(lat>=wj_area_S) & (lat<=wj_area_N)]
lon_wj_range = lon[(lon>=wj_area_W) & (lon<=wj_area_E)]
ERA5_wj_axis = ca.cal_ridge_line(uERA5_ver_JJA.sel(level=200.0, lat=lat_wj_range, lon=lon_wj_range).mean(dim="time", skipna=True), ridge_trough="max")
his_wj_axis_lat = np.zeros((26, 41))
his_wj_axis_lon = np.zeros((26, 41))
ssp585_p3_wj_axis_lat = np.zeros((26, 41))
ssp585_p3_wj_axis_lon = np.zeros((26, 41))
for num_mod, mod in enumerate(models_array):
    his_wj_axis_lat[num_mod,:], his_wj_axis_lon[num_mod,:] = ca.cal_ridge_line(uhis_ver_JJA.sel(level=200.0, lat=lat_wj_range, lon=lon_wj_range, models=mod).mean(dim="time", skipna=True), ridge_trough="max")
    ssp585_p3_wj_axis_lat[num_mod,:], ssp585_p3_wj_axis_lon[num_mod,:] = ca.cal_ridge_line(ussp585_p3_ver_JJA.sel(level=200.0, lat=lat_wj_range, lon=lon_wj_range, models=mod).mean(dim="time", skipna=True), ridge_trough="max")
his_wj_axis = xr.DataArray(
    data=his_wj_axis_lat,
    dims=["models", "lon"],
    coords=dict(
        models=(["models"], models_array),
        lon=(["lon"], lon_wj_range.data)
    )
)
ssp585_p3_wj_axis = xr.DataArray(
    data=ssp585_p3_wj_axis_lat,
    dims=["models", "lon"],
    coords=dict(
        models=(["models"], models_array),
        lon=(["lon"], lon_wj_range.data)
    )
)
his_wj_axis_ens = his_wj_axis.mean(dim="models", skipna=True)
ssp585_p3_wj_axis_ens = ssp585_p3_wj_axis.mean(dim="models", skipna=True)

his_wj_axis_lat_gens = his_wj_axis.sel(models=gmodels).mean(dim="models", skipna=True)
ssp585_p3_wj_axis_lat_gens = ssp585_p3_wj_axis.sel(models=gmodels).mean(dim="models", skipna=True)

#   calculate the good models difference between historical run and ssp585_p3 run
pre_diff_India_pre_slope = pre_ssp585_p3_India_pre_slope - pre_his_India_pre_slope

pre_diff_India_pre_slope_ens = pre_diff_India_pre_slope.mean(dim="models",skipna=True)
pre_diff_India_pre_slope_ens_mask = ca.cal_mmemask(pre_diff_India_pre_slope)

pre_diff_India_pre_slope_gens = pre_diff_India_pre_slope.sel(models=gmodels).mean(dim="models", skipna=True)

pre_diff_India_pre_gens_mask = ca.cal_mmemask(pre_diff_India_pre_slope.sel(models=gmodels))

pre_diff_India_pre_rvalue = ca.cal_rdiff(pre_ssp585_p3_India_pre_rvalue, pre_his_India_pre_rvalue)
pre_diff_India_pre_rvalue_ens = ca.cal_rMME(pre_diff_India_pre_rvalue, "models")
pre_diff_India_pre_rvalue_ens_mask = ca.cal_mmemask(pre_diff_India_pre_rvalue)

pre_diff_India_pre_rvalue_gens = ca.cal_rMME(pre_diff_India_pre_rvalue.sel(models=gmodels), "models")

IndR_diff_hgt_slope = IndR_ssp585_p3_hgt_slope - IndR_his_hgt_slope
IndR_diff_hgt_slope_ens = IndR_diff_hgt_slope.mean(dim="models", skipna=True)
IndR_diff_hgt_slope_gens = IndR_diff_hgt_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_hgt_ens_mask = ca.cal_mmemask(IndR_diff_hgt_slope)
IndR_diff_hgt_gens_mask = ca.cal_mmemask(IndR_diff_hgt_slope.sel(models=gmodels))

IndR_diff_sst_slope = IndR_ssp585_p3_sst_slope - IndR_his_sst_slope
IndR_diff_sst_slope_ens = IndR_diff_sst_slope.mean(dim="models", skipna=True)
IndR_diff_sst_slope_gens = IndR_diff_sst_slope.sel(models=gmodels).mean(dim="models", skipna=True)

IndR_diff_sst_ens_mask = ca.cal_mmemask(IndR_diff_sst_slope)
IndR_diff_sst_gens_mask = ca.cal_mmemask(IndR_diff_sst_slope.sel(models=gmodels))

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
IndR_diff_hgt_rvalue_ens = ca.cal_rMME(IndR_diff_hgt_rvalue, "models")
IndR_diff_hgt_rvalue_gens = ca.cal_rMME(IndR_diff_hgt_rvalue.sel(models=gmodels), "models")

IndR_diff_sst_rvalue = ca.cal_rdiff(IndR_ssp585_p3_sst_rvalue, IndR_his_sst_rvalue)
IndR_diff_sst_rvalue_gens = ca.cal_rMME(IndR_diff_sst_rvalue.sel(models=gmodels), "models")
IndR_diff_sst_rvalue_ens = ca.cal_rMME(IndR_diff_sst_rvalue, "models")


IndR_diff_u_rvalue = ca.cal_rdiff(IndR_ssp585_p3_u_rvalue, IndR_his_u_rvalue)
IndR_diff_u_rvalue_ens = ca.cal_rMME(IndR_diff_u_rvalue, "models")
IndR_diff_u_rvalue_gens = ca.cal_rMME(IndR_diff_u_rvalue.sel(models=gmodels), "models")

IndR_diff_v_rvalue = ca.cal_rdiff(IndR_ssp585_p3_v_rvalue, IndR_his_v_rvalue)
IndR_diff_v_rvalue_ens = ca.cal_rMME(IndR_diff_v_rvalue, "models")
IndR_diff_v_rvalue_gens = ca.cal_rMME(IndR_diff_v_rvalue.sel(models=gmodels), "models")

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

# calculate the precipitation standard deviation for ERA5, historical and ssp585_p3 and then calculate the difference between ssp585_p3 and historical
preGPCP_JJA_std = preGPCP_JJA.std(dim="time",skipna=True)
prehis_JJA_std = prehis_JJA.std(dim="time",skipna=True)
pressp585_p3_JJA_std = pressp585_p3_JJA.std(dim="time",skipna=True)

prediff_JJA_std = pressp585_p3_JJA_std - prehis_JJA_std

prehis_JJA_std_ens = prehis_JJA_std.mean(dim="models",skipna=True)
pressp585_p3_JJA_std_ens = pressp585_p3_JJA_std.mean(dim="models",skipna=True)
prediff_JJA_std_ens = prediff_JJA_std.mean(dim="models",skipna=True)

# read the linear trend of different variables such as precipitation, hgt, u, v and SST
pretrend = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/preCMIP6_trend.nc")
hgttrend = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/hgtCMIP6_trend.nc")
utrend = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/uCMIP6_trend.nc")
vtrend = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/vCMIP6_trend.nc")
tostrend = xr.open_dataarray("/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/CMIP6/ssp585/tmp_var/JJA/non_detrend/tosCMIP6_trend.nc")

pretrend_ens = pretrend.mean(dim="models",skipna=True)
hgttrend_ens = hgttrend.mean(dim="models",skipna=True)
utrend_ens = utrend.mean(dim="models",skipna=True)
vtrend_ens = vtrend.mean(dim="models",skipna=True)
tostrend_ens = tostrend.mean(dim="models",skipna=True)

# %% ##mark: plot the figure 1: precipitation regress onto SASMR, observation and MME, 1979-2014, regression coefficient
pplt.rc.grid = False
pplt.rc.reso = "lo"
pplt.rc["figure.facecolor"] = "white"
pplt.rc["font.large"]
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 3), (2, 1))
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([60, 90, 120])  # 设置纬度刻度
yticks = np.arange(0, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [50.0, 140.0, yticks[0], 45.0]
sepl.geo_ticks(axs, xticks, yticks, cl, extents, majorticklabelsize=10)

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
    sepl.patches(ax, x0 - cl, y0, width, height, proj, linestyle="-", linewidth=1.2)
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj, linewidth=1.2)
# ===================================================
con = axs[0].contourf(
    pre_AIR_India_pre_slope,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.6,1.7,0.2),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_AIR_India_pre_slope, axs[0], n, np.where(pre_AIR_India_pre_pvalue[::n, ::n] < 0.10), "bright purple", 12.0,
)

axs[0].format(
    ltitle="1979-2014", rtitle="GPCP & AIR",
)
# ===================================================
con = axs[1].contourf(
    pre_his_India_pre_slope_ens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.6,1.7,0.2),
    zorder=0.8,
    extend="both",
    )
sepl.plt_sig(
    pre_his_India_pre_slope_ens, axs[1], n, np.where(pre_his_India_pre_slope_ens_mask[::n, ::n] > 0.00), "bright purple", 12.0,
)

axs[1].format(
    ltitle="1979-2014", rtitle="MME",
)
# ===================================================
cb = fig.colorbar(con, loc="b", width=0.13, length=0.7, label="", ticklabelsize=8)
cb.set_ticks(np.arange(-2.0,2.1, 0.4))
axs.format(linewidth=1.2)
fig.format(abc="(a)", abcloc="l")
# %% ##mark: plot the figure 2: circulation regress onto SASMR, observation and MME, 1979-2014, 200/500/850hPa
startlevel=[-15, -8, -6]
spacinglevel=[1.5, 0.8, 0.6]
scalelevel=[0.30, 0.17, 0.14]

pplt.rc.grid = False
pplt.rc.reso = "lo"
pplt.rc["figure.facecolor"] = "white"
pplt.rc["font.large"] = 15
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=6.0, wspace=6.0, hspace=5.5, outerpad=2.0)
plot_array = np.reshape(range(1, 7), (3, 2))
# plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([60, 90, 120])  # 设置纬度刻度
yticks = np.arange(0, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [40.0, 140.0, yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, extents, majorticklabelsize=14, lonminorspace=5, coastlinewidth=1.7, majorticklabelpad=3.0, majorticklen=6.0, minorticklen=5.0)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ======================================
for ax in axs:
    rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=1.3, zorder=1.1)
    ax.add_patch(rect)
    # # India area
    # x0 = India_W
    # y0 = India_S
    # width = India_E-India_W
    # height = India_N-India_S
    # sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # # NC area
    # x0 = NC_W
    # y0 = NC_S
    # width = NC_E-NC_W
    # height = NC_N-NC_S
    # sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # # SJ-KP area
    # x0 = SJ_W
    # y0 = SJ_S
    # width = SJ_E-SJ_W
    # height = SJ_N-SJ_S
    # sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ======================================
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    if lev == 200.0:
        for ax in axs[num_lev, :]:
            x0 = lon_ranking1_W
            y0 = lat_ranking1_S
            width = lon_ranking1_E-lon_ranking1_W
            height = lat_ranking1_N-lat_ranking1_S
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-", linewidth=1.7)
            # #   EAM index
            # x0 = 110.0
            # y0 = 40.0
            # width = 40.0
            # height = 10.0
            # sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="green8", linestyle="-")
            # x0 = 110.0
            # y0 = 25.0
            # width = 40.0
            # height = 10.0
            # sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="green8", linestyle="-")
    elif lev == 850.0:
        for ax in axs[num_lev, :]:
            x0 = lon_ranking2_W
            y0 = lat_ranking2_S
            width = lon_ranking2_E-lon_ranking2_W
            height = lat_ranking2_N-lat_ranking2_S
            sepl.patches(ax, x0 - cl, y0, width, height, proj, edgecolor="bright purple", linestyle="-", linewidth=1.7)
    con = axs[num_lev, 0].contourf(
        IndRAIR_ERA5_hgt_slope.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndRAIR_ERA5_hgt_slope.sel(level=lev), axs[num_lev, 0], n, np.where(IndRAIR_ERA5_hgt_pvalue.sel(level=lev)[::n, ::n] <= 0.05), "bright purple", 19.0,
    )
    # axs[num_lev, 0].quiver(
    #     IndRAIR_ERA5_u_slope.sel(level=lev)[::ski, ::ski],
    #     IndRAIR_ERA5_v_slope.sel(level=lev)[::ski, ::ski],
    #     zorder=1.1,
    #     headwidth=2.6,
    #     headlength=2.3,
    #     headaxislength=2.3,
    #     scale_units="xy",
    #     scale=scalelevel[num_lev],
    #     pivot="mid",
    #     color="grey6",
    # )

    m = axs[num_lev, 0].quiver(
        IndRAIR_ERA5_u_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndRAIR_ERA5_v_slope.sel(level=lev).where(IndRAIR_ERA5_wind_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        minlength=0.5,
        width=0.004,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 0].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.10, fontproperties={"size": 10}, zorder=3.1,
    )
    axs[num_lev, 0].format(
        ltitle="1979-2014 {:.0f}hPa".format(lev), rtitle="AIR & ERA5",
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
        IndR_his_hgt_slope_ens.sel(level=lev), axs[num_lev, 1], n, np.where(IndR_his_hgt_slope_ens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 19.0,
    )
    # axs[num_lev, 1].quiver(
    #     IndR_his_u_slope_ens.sel(level=lev)[::ski, ::ski],
    #     IndR_his_v_slope_ens.sel(level=lev)[::ski, ::ski],
    #     zorder=1.1,
    #     headwidth=2.6,
    #     headlength=2.3,
    #     headaxislength=2.3,
    #     scale_units="xy",
    #     scale=scalelevel[num_lev],
    #     pivot="mid",
    #     color="grey6",
    # )

    m = axs[num_lev, 1].quiver(
        IndR_his_u_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        IndR_his_v_slope_ens.sel(level=lev).where(IndR_his_wind_ens_mask.sel(level=lev) > 0.0)[::ski, ::ski],
        zorder=1.1,
        headwidth=2.6,
        headlength=2.3,
        headaxislength=2.3,
        minlength=0.5,
        width=0.004,
        scale_units="xy",
        scale=scalelevel[num_lev],
        pivot="mid",
        color="black",
    )

    qk = axs[num_lev, 1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.10, fontproperties={"size": 10}, zorder=3.1,
    )
    axs[num_lev, 1].format(
        ltitle="1979-2014 {:.0f}hPa".format(lev), rtitle="MME",
    )
    cb = axs[num_lev, 1].colorbar(con, loc="r", width=0.17, length=0.8, label="", ticklabelsize=12, pad=2.5)
    cb.set_ticks(np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]*2))
# ======================================
axs.format(linewidth=1.2, titlepad=7.0)
fig.format(abc="(a)", abcloc="l")
# %% ##mark: plot the figure 3:  taylor-diagram including 200hPa pcc and 850hPa pcc
labels = list(models.data)
labels.append("MME")
labels.append("gMME")
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#绘图
fig=plt.figure(figsize=(13,8),dpi=300)
plt.rc('font',size=15)
plt.rcParams["axes.facecolor"] = "white"

#调用函数
ax1=fig.add_subplot(111,projection='polar')
box = ax1.get_position()
ax1.set_position([0, box.y0, box.width*1.2, box.height])
# ax1.text(0.6,0.1,'(a)',fontsize=15)
# tar(ax1,np.array(IndR_EAM_pcc),np.array(IndR_EAM_std),labels)
sepl.taylor_diagram(ax1,np.array(IndR_200hgt_pcc),np.array(IndR_200hgt_std), dotlables=labels, lables=True, color="r")
sepl.taylor_diagram(ax1,np.array(IndR_850hgt_pcc),np.array(IndR_850hgt_std), color="b")
plt.legend(loc="center left", bbox_to_anchor=(1.1,0.5), ncol=2, frameon=True, numpoints=1, handlelength=0)
# %% ##mark: plot the figure 4:  precipitation regress onto SASMR, gMME, 1979-2014 & 2064-2099 & diff
pplt.rc.grid = False
pplt.rc.reso = "lo"
pplt.rc["figure.facecolor"] = "white"
pplt.rc["font.large"] = 11
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 4), (3, 1))
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([60, 90, 120])  # 设置经度刻度
yticks = np.arange(0, 46, 15)  # 设置纬度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [50.0, 140.0, yticks[0], 45.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 5, extents, labelsize=10)

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
    sepl.patches(ax, x0 - cl, y0, width, height, proj, linestyle="-")
    # NC area
    x0 = NC_W
    y0 = NC_S
    width = NC_E-NC_W
    height = NC_N-NC_S
    sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ===================================================
con = axs[0].contourf(
    pre_his_India_pre_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.6,1.7,0.2),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_his_India_pre_slope_gens, axs[0], n, np.where(pre_his_India_pre_slope_gens_mask[::n, ::n] > 0.00), "bright purple", 12.0,
)

axs[0].format(
    ltitle="1979-2014", rtitle="gMME",
)
# ===================================================
con = axs[1].contourf(
    pre_ssp585_p3_India_pre_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.6,1.7,0.2),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_ssp585_p3_India_pre_slope_gens, axs[1], n, np.where(pre_ssp585_p3_India_pre_slope_gens_mask[::n, ::n] > 0.00), "bright purple", 12.0,
)

axs[1].format(
    ltitle="2064-2099", rtitle="gMME",
)
# ===================================================
cb = axs[1].colorbar(con, loc="b", width=0.13, length=0.85, label="", ticklabelsize=9)
cb.set_ticks(np.arange(-2.0,2.1, 0.4))
# ===================================================
con = axs[2].contourf(
    pre_diff_India_pre_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94},
    levels=np.arange(-1.2,1.3,0.2),
    zorder=0.8,
    extend="both"
    )
sepl.plt_sig(
    pre_diff_India_pre_slope_gens, axs[2], n, np.where(pre_diff_India_pre_gens_mask[::n, ::n] > 0.00), "bright purple", 12.0,
)

axs[2].format(
    ltitle="diff", rtitle="gMME",
)
# ===================================================
cb = fig.colorbar(con, loc="b", width=0.13, length=0.85, label="", ticklabelsize=9)
cb.set_ticks(np.arange(-1.2, 1.3, 0.4))
fig.format(abc="(a)", abcloc="l")
# %%
# %% ##mark: plot the figure 5:  circulation regress onto SASMR, gMME, 1979-2014 & 2064-2099 & diff, 200/500/850hPa
startlevel=[-15, -8, -6]
spacinglevel=[1.5, 0.8, 0.6]
scalelevel=[0.23, 0.17, 0.14]

pplt.rc.grid = False
pplt.rc.reso = "lo"
pplt.rc["figure.facecolor"] = "white"
pplt.rc["font.large"] = 16
cl = 0  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=6.0, wspace=4.2, hspace=3.7, outerpad=2.0)
plot_array = np.reshape(range(1, 10), (3, 3))
# plot_array[-1,-1] = 0
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([60, 90, 120])  # 设置纬度刻度
yticks = np.arange(0, 46, 15)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [40.0, 140.0, yticks[0], 55.0]
sepl.geo_ticks(axs, xticks, yticks, cl, 5, 5, extents, labelsize=14, marjotticklen=6.0, minorticklen=5.0)
# ===================================================
ski = 2
n = 1
w, h = 0.12, 0.14
# ======================================
for ax in axs:
    rect = Rectangle((1 - w, 0), w, h, transform=ax.transAxes, fc="white", ec="k", lw=0.5, zorder=1.1)
    ax.add_patch(rect)
    # # India area
    # x0 = India_W
    # y0 = India_S
    # width = India_E-India_W
    # height = India_N-India_S
    # sepl.patches(ax, x0 - cl, y0, width, height, proj, linestyle="-")
    # # NC area
    # x0 = NC_W
    # y0 = NC_S
    # width = NC_E-NC_W
    # height = NC_N-NC_S
    # sepl.patches(ax, x0 - cl, y0, width, height, proj)
    # # SJ-KP area
    # x0 = SJ_W
    # y0 = SJ_S
    # width = SJ_E-SJ_W
    # height = SJ_N-SJ_S
    # sepl.patches(ax, x0 - cl, y0, width, height, proj)
# ======================================
for num_lev,lev in enumerate([200.0, 500.0, 850.0]):
    con = axs[num_lev, 0].contourf(
        IndR_his_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_his_hgt_slope_gens.sel(level=lev), axs[num_lev, 0], n, np.where(IndR_his_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 12.0,
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
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.10, fontproperties={"size":10}, zorder=3.1,
    )
    axs[num_lev, 0].format(
        ltitle="1979-2014 {:.0f}hPa".format(lev), rtitle="gMME",
    )
	# ======================================
    con = axs[num_lev, 1].contourf(
        IndR_ssp585_p3_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_ssp585_p3_hgt_slope_gens.sel(level=lev), axs[num_lev, 1], n, np.where(IndR_ssp585_p3_hgt_slope_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 12.0,
    )
    axs[num_lev, 1].quiver(
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

    m = axs[num_lev, 1].quiver(
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

    qk = axs[num_lev, 1].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.10, fontproperties={"size":10}, zorder=3.1,
    )
    axs[num_lev, 1].format(
        ltitle="2064-2099 {:.0f}hPa".format(lev), rtitle="gMME",
    )
# ======================================
    con = axs[num_lev, 2].contourf(
        IndR_diff_hgt_slope_gens.sel(level=lev),
        cmap="ColdHot",
        cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
        levels=np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]),
        zorder=0.8,
        extend="both"
    )
    sepl.plt_sig(
        IndR_diff_hgt_slope_gens.sel(level=lev), axs[num_lev, 2], n, np.where(IndR_diff_hgt_gens_mask.sel(level=lev)[::n, ::n] > 0.00), "bright purple", 12.0,
    )
    axs[num_lev, 2].quiver(
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

    m = axs[num_lev, 2].quiver(
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

    qk = axs[num_lev, 2].quiverkey(
        m, X=1 - w / 2, Y=0.7 * h, U=0.5, label="0.5", labelpos="S", labelsep=0.10, fontproperties={"size":10}, zorder=3.1,
    )
    axs[num_lev, 2].format(
        ltitle="diff {:.0f}hPa".format(lev), rtitle="gMME",
    )
    cb = axs[num_lev, 2].colorbar(con, loc="r", width=0.15, length=0.8, label="", ticklabelsize=13, pad=2.0)
    cb.set_ticks(np.arange(startlevel[num_lev], -startlevel[num_lev]+spacinglevel[num_lev], spacinglevel[num_lev]*2))
# ======================================
fig.format(abc="(a)", abcloc="l")
# %% ##mark: plot the figure 6:  bar plots that show the WAAC, EAAC and WNPAC in all 26 models, gMME and MME in the period of 1979-2014 & 2064-2099 & diff
pplt.rc["figure.facecolor"] = "white"
pplt.rc["font.large"] = 12
pplt.rc["legend.fontsize"] = 10
fig = pplt.figure(span=False, share=False, refheight=3.0, refwidth=10.0, wspace=4.0, hspace=6.0, outerpad=2.0)
axs = fig.subplots(ncols=1, nrows=3)

#   plot the bar-plot of the WA high
plot_data = np.zeros((28,3))
plot_data[:-2,0] = IndR_his_WAhigh_regress[0].data
plot_data[:-2,1] = IndR_ssp585_p3_WAhigh_regress[0].data
plot_data[:-2,2] = IndR_diff_WAhigh_slope.data
plot_data[-2,0] = IndR_his_WAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-2,1] = IndR_ssp585_p3_WAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-2,2] = IndR_diff_WAhigh_slope.sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,0] = IndR_his_WAhigh_regress[0].mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_WAhigh_regress[0].mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_WAhigh_slope.mean(dim="models", skipna=True).data
plt.rc("text", usetex=False)
label_models = [r"$\bf{"+i+"}$" if i in gmodels else i for i in models_array]
label_models.append("gMME")
label_models.append("MME")

m = axs[0].bar(label_models,plot_data*1e6,width=0.5,cycle="tab10",edgecolor="grey7")

axs[0].axhline(0,lw=1.5,color="grey7")
# axs[0].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[0].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[0].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[0].legend(handles=m, loc='ur', labels=["1979-2014", "2064-2099", "diff"])
axs[0].format(ylim=(-3.0,3.0),xlocator=np.arange(0,28), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="WA")

#   plot the bar-plot of the EA high
plot_data = np.zeros((28,3))
plot_data[:-2,0] = IndR_his_EAhigh_regress[0].data
plot_data[:-2,1] = IndR_ssp585_p3_EAhigh_regress[0].data
plot_data[:-2,2] = IndR_diff_EAhigh_slope.data
plot_data[-2,0] = IndR_his_EAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-2,1] = IndR_ssp585_p3_EAhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-2,2] = IndR_diff_EAhigh_slope.sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,0] = IndR_his_EAhigh_regress[0].mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_EAhigh_regress[0].mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_EAhigh_slope.mean(dim="models", skipna=True).data


m = axs[1].bar(label_models,plot_data*1e6,width=0.5,cycle="tab10",edgecolor="grey7")
axs[1].axhline(0,lw=1.5,color="grey7")

# axs[1].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[1].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[1].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[1].legend(handles=m, loc='ur', labels=["1979-2014", "2064-2099", "diff"])
axs[1].format(ylim=(-2.0,2.0),xlocator=np.arange(0,28), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="EA")

#   plot the bar-plot of the WNP high
plot_data = np.zeros((28,3))
plot_data[:-2,0] = IndR_his_WNPhigh_regress[0].data
plot_data[:-2,1] = IndR_ssp585_p3_WNPhigh_regress[0].data
plot_data[:-2,2] = IndR_diff_WNPhigh_slope.data
plot_data[-2,0] = IndR_his_WNPhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-2,1] = IndR_ssp585_p3_WNPhigh_regress[0].sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-2,2] = IndR_diff_WNPhigh_slope.sel(models=gmodels).mean(dim="models", skipna=True).data
plot_data[-1,0] = IndR_his_WNPhigh_regress[0].mean(dim="models", skipna=True).data
plot_data[-1,1] = IndR_ssp585_p3_WNPhigh_regress[0].mean(dim="models", skipna=True).data
plot_data[-1,2] = IndR_diff_WNPhigh_slope.mean(dim="models", skipna=True).data


m = axs[2].bar(label_models,plot_data*1e6,width=0.5,cycle="tab10",edgecolor="grey7")
axs[2].axhline(0,lw=1.5,color="grey7")
# axs[2].axhline(ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# axs[2].axhline(-ca.cal_rlim1(0.95, 36),lw=1.5,color="grey7",ls='--')
# for num,i in enumerate(gmodels):
#     if i > 0:
#         axs[2].plot(num, 0, marker='o', markersize=8,zorder=100, color="red")

axs[2].legend(handles=m, loc='ur', labels=["1979-2014", "2064-2099", "diff"])
axs[2].format(ylim=(-1.2,1.2),xlocator=np.arange(0,28), xtickminor=False, ytickminor=False, grid=False, tickwidth=1.5, ticklen=6.0, linewidth=1.5, edgecolor="grey8", rtitle="WNP")

axs.format(xrotation=45, ticklabelsize=10.0)
fig.format(abc="(a)", abcloc="l")
# %% ##mark: plot the figure 7:  Sea surface temperature regress onto SASMR, observation and gMME, 1979-2014 & 2064-2099 & diff
startlevel=-6e-1
spacinglevel=0.06
pplt.rc["figure.facecolor"] = "white"
pplt.rc["font.large"] = 10
pplt.rc["legend.fontsize"] = 10
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180  # 设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)

fig = pplt.figure(span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0)
plot_array = np.reshape(range(1, 5), (4, 1))
axs = fig.subplots(plot_array, proj=proj)

#   set the geo_ticks and map projection to the plots
# xticks = np.array([30, 60, 90, 120, 150, 180])  # 设置纬度刻度
xticks = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300])  # 设置纬度刻度
yticks = np.arange(0, 61, 30)  # 设置经度刻度
# 设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
# 当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [30, 300, -15, yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, 10, 10, extents, labelsize=8)
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
    IndR_his_sst_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_his_sst_slope_gens, axs[1], n, np.where(IndR_his_sst_slope_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[1].format(
    rtitle="1979-2014", ltitle="gMME",
)
# ======================================
con = axs[2].contourf(
    IndR_ssp585_p3_sst_slope_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
) 
sepl.plt_sig(
    IndR_ssp585_p3_sst_slope_gens, axs[2], n, np.where(IndR_ssp585_p3_sst_slope_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[2].format(
    rtitle="2064-2099", ltitle="gMME",
)
cb = axs[2].colorbar(con, loc="b", width=0.13, length=0.7, label="")
cb.set_ticks(np.arange(-0.6,0.61,0.24))
# ======================================
startlevel=-1.0
spacinglevel=0.1
# ======================================
con = axs[3].contourf(
    IndR_diff_sst_rvalue_gens,
    cmap="ColdHot",
    cmap_kw={"left": 0.06, "right": 0.94, "cut": -0.1},
    levels=np.arange(startlevel, -startlevel+spacinglevel, spacinglevel),
    zorder=0.8,
    extend="both"
)
sepl.plt_sig(
    IndR_diff_sst_rvalue_gens, axs[3], n, np.where(IndR_diff_sst_gens_mask[::n, ::n] > 0.00), "bright purple", 3.0,
)
axs[3].format(
    rtitle="diff", ltitle="gMME",
)
cb = axs[3].colorbar(con, loc="b", width=0.13, length=0.7, label="")
cb.set_ticks(np.arange(-1.0,1.1,0.4))
# ======================================
fig.format(abc="(a)", abcloc="l")

# %%
pplt.rc.grid = False
pplt.rc.reso = "lo"
cl = 180	#设置地图投影的中心纬度
proj = pplt.PlateCarree(central_longitude=cl)
array = [[1, 2, 3, 4], [0, 5, 6, 7], [0, 0, 8, 9], [0, 0, 0, 10]]

fig = pplt.figure(
    span=False, share=False, refwidth=4.0, wspace=4.0, hspace=3.5, outerpad=2.0
)
axs = fig.subplots(array, proj=proj)

#   set the geo_ticks and map projection to the plots
xticks = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300])	#设置纬度刻度
yticks = np.arange(-30, 46, 15)						#设置经度刻度
#设置绘图的经纬度范围extents，其中前两个参数为经度的最小值和最大值，后两个数为纬度的最小值和最大值
#当想要显示的经纬度范围不是正好等于刻度显示范围时，对extents进行相应的修改即可
extents = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
sepl.geo_ticks(axs, xticks, yticks, cl, extents)
# %%
